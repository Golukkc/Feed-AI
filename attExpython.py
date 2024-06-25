import pandas as pd
import requests
from urllib.parse import urlparse
# from PIL import Image
import google.generativeai as genai
# import google.ai.generativelanguage as glm
# from collections import Counter
# import base64
# import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.cloud import aiplatform

import google.generativeai as genai

# from google.generativeai.types import ContentType

# from IPython.display import Markdown
import time
# import cv2
import os
# from collections import Counter
# import re

GOOGLE_API_KEY = "AIzaSyBEUCyDsA9R8TFqWlinhVmF4phm6TsITMQ"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Vertex AI client with your project ID and location
aiplatform.init(project="stellar-display-145814", location="us-central1")
choose=GenerativeModel("gemini-1.5-flash")

def filter_image_columns(df):
    # Identify columns related to images
    image_columns = [col for col in df.columns if 'image' in col.lower() or 'img' in col.lower()]
    
    # Keep only those image columns that are not completely empty
    non_empty_image_columns = [col for col in image_columns if df[col].notna().any()]
    
    # Combine non-image columns with non-empty image columns
    columns_to_keep = [col for col in df.columns if col not in image_columns] + non_empty_image_columns
    
    # Filter the dataframe
    filtered_df = df[columns_to_keep]
    
    return filtered_df

def process_additional_image_links(df, column_name):
    # Split the 'additional_image_link' column into multiple columns
    additional_image_links = df[column_name].str.split(',', expand=True)
    
    # Rename the new columns
    additional_image_links.columns = [f'image_{i+1}' for i in range(additional_image_links.shape[1])]
    
    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df, additional_image_links], axis=1)
    
    return df

def rename_and_append_image_columns(df):
    # Find all columns matching the pattern 'image[0].url', 'image[1].url', etc.
    image_pattern_columns = [col for col in df.columns if col.startswith('image[') and col.endswith('].url')]
    
    # Determine the next index for additional image links
    existing_additional_columns = [col for col in df.columns if col.startswith('image_')]
    next_index = len(existing_additional_columns) + 1
    
    # Rename the columns and append them to the DataFrame
    renamed_columns = {}
    for col in image_pattern_columns:
        new_col_name = f'additional_image_link_{next_index}'
        renamed_columns[col] = new_col_name
        next_index += 1
    
    # Rename the columns in the DataFrame
    df = df.rename(columns=renamed_columns)
    return df

# Function to download an image from URL to a local file
def download_image(url, filename):
    retries = 3  # Number of retry attempts
    retry_delay = 2  # Delay between retries in seconds

    for _ in range(retries):
        try:
            response = requests.get(url, timeout=10)  # Adjust timeout as needed
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                # Check file size
                if os.path.getsize(filename) == int(response.headers.get('Content-Length', 0)):
                    return filename
                else:
                    print(f"Downloaded file size mismatch for {url}. Retrying...")
                    continue
            else:
                print(f"Failed to download image from {url}. Status code: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Error downloading image from {url}: {e}")
            time.sleep(retry_delay)
            continue

    print(f"Failed to download image from {url} after {retries} retries.")
    return None

def process_images_and_generate_content(url_column, productName, description,productCategoryList, tags, more_info, Brand, product_type,category,generation_config, safety_settings):
    prompt = f"""
Analyze the provided images, title, description, and other attributes of the {category} of the brand named {Brand}. Apply your intelligence to identify and predict the following attributes in an easy-to-read Excel format. Ensure that each attribute has a predicted value. If you can't find a value from the text, use the images to fill in each attribute's value. If the value still can't be determined, use your best judgment to provide a reasonable value.
Input:
Title: {productName}
Description: {description}
productCategoryList: {productCategoryList}
Tags: {tags}
more_info: {more_info}


Output Format:
| Attribute             | Predicted Value |
|-----------------------|-----------------|
| Brand                 |   {Brand}       |
| Gender                |                 |
| Material              |                 |
| Product Type          |                 |
| Product Category      | {product_type}  |
| Color                 |                 |
| Print                 |                 |
| Fit                   |                 |
| Sleeve Type           |                 |
| Collar Type           |                 |
| Cuff Style            |                 |
| Product USPs          |                 |
| Pocket Type           |                 |
| Logo Details          |                 |
| Marketplace           |     Iconic      |

Attributes to Look for:

Brand: Identify the brand of the shirt. Use the given {Brand} value.
Gender: Determine the intended demographic for the shirt, specifying whether it is targeted towards Men or Women preferences. This can be inferred from the provided text or visual cues from images. If it is for Kids, Give Unisex Kids. 
Material: Identify the fabric or material of the shirt, such as 100% Cotton, Viscose, Crepe, Georgette, Rayon, Satin, Linen, Nylon, Knitted, Chiffon, Silk, Woolen, Sequin, Denim, etc. This can be found in the provided text or packaging details in images. If the Material cannot be inferred directly from the text description, analyze the images for any relevant symbols or indications. If still not found in the images, use overall knowledge from the text description and images to make an educated prediction. Always provide a predicted value rather than leaving the attribute blank or using placeholders like "none," "not specified," or "N/A. Try not to give Polyster
Shirt Type: Determine the type of shirt, such as Casual, Formal, or Party etc. based on the provided text and images.
Product Category: Confirm it is a Shirt.
Color: Identify the color of the shirt from the description or images. Example values- Yellow, Black, Brown, Grey, Maroon, Blue, Green, Red, Pink, Beige, etc.
Print: Identify the print pattern of the shirt, such as Solid, Striped, Check, Textured, or Printed etc. If Checks, Checkered or Tartan Check, Give Checked 
Sleeve Type: Identify the type of sleeves the shirt has, such as Full Sleeve, Half Sleeve, etc. If it is Long Sleeve or Full Sleeve, Give Full Sleeve.
Collar Type: Identify the type of collar on the shirt, such as Button-Down Collar, Chinese Collar, etc. If Button-Down Collar or Button-Down, Give Button-Down Collar and For Classic Shirt Collar give Classic Collar
Cuff Style: Determine the style of the cuffs on the shirt, such as Convertible Cuff, French Cuff, Barrel Cuff, Rounded Cuff, Cocktail Cuff, etc.If Standard, Give Standard Cuff. If the Cuff Style cannot be inferred directly from the text description, analyze the images for any relevant symbols or indications. If still not found in the images, use overall knowledge from the text description and images to make an educated prediction. Always provide a predicted value rather than leaving the attribute blank or using placeholders like "none," "not specified," or "N/A."
Product USPs: Identify any unique selling points or special features of the shirt, such as Wrinkle Free, Iron Free, etc. If the Product USPs cannot be inferred directly from the text description, analyze the images for any relevant symbols or indications. If still not found in the images, use overall knowledge from the text description and images to make an educated prediction. Always provide a predicted value rather than leaving the attribute blank or using placeholders like "None," "not specified," or "N/A. Try not to give Polyster"
Pocket Type: Identify the type of pockets the shirt has, such as Double Pocket, Single Pocket etc. .If no pocket give None. If the Pocket Type cannot be inferred directly from the text description, analyze the images for any relevant symbols or indications. If still not found in the images, use overall knowledge from the text description and images to make an educated prediction. Always provide a predicted value rather than leaving the attribute blank or using placeholders like "none," "not specified," or "N/A."
Logo Details: Identify any details about logos on the shirt, such as an Embroidered Logo. If no logo, leave blank.
Marketplace: This is fixed as Iconic.
Please ensure that all attributes have predicted values based on available information from both textual descriptions and visual analysis of images. If specific details are not explicitly stated, use reasonable inference based on the overall context provided. This approach ensures comprehensive attribute prediction for the perfume product.

Example:

Title: "GANT YELLOW BROADCLOTH REGULAR FIT SHIRT"
Description: "Our iconic GANT Men Yellow Solid Collar Shirt is crafted from 100% cotton broadcloth for a classic, timeless look. Featuring our signature buttondown collar, locker loop and boxpleat at the back, this regular fit shirt is a must-have. Complete your look with our classic shield logo embroidered at the chest. This bestselling shirt is sure to become an iconic wardrobe staple."


Output Format:
| Attribute             | Predicted Value |
|-----------------------|-----------------|
| Brand                 |   Gant          |
| Gender                |     Men         |
| Material              |   100% Cotton   |
| Product Type          |     Casual      |
| Product Category      |   Shirt         |
| Color                 |    Yellow       |
| Print                 |    Solid        |
| Fit                   |   Regular Fit   |
| Sleeve Type           |    Full Sleeve  |
| Collar Type           |Button-Down Collar|
| Cuff Style            |Convertible Cuff |
| Product USPs          |  Wrinkle Free   |
| Pocket Type           |  Double Pocket  |
| Logo Details          |Embroidered Logo |
| Marketplace           |   Iconic        |
    """

    file_names = []
    for url in url_column:
        filename = download_image(url, "temp_image.jpg")
        if filename:
            try:
                # Upload the file using genai.upload_file
                print(f"Uploading file: {filename}")
                file_response = genai.upload_file(filename)
                print("Upload response:", file_response)

                # Extract the file name
                if hasattr(file_response, 'name'):
                    file_names.append(file_response.name)
                else:
                    print(f"Failed to upload file {filename} to GenAI. File response: {file_response}")
                    continue
                
            except Exception as e:
                print(f"Error uploading file {filename} to GenAI: {e}")
                continue  # Skip to the next image if upload fails
            # Add file references to the prompt
    for file_name in file_names:
        prompt += f"\nFile: {file_name}\n"

    try:
        print("Generating content...")
        model = choose  # Ensure this is correctly initialized with your model
        # generation_config["maxOutputTokens"] = min(generation_config.get("maxOutputTokens", 2048), 2048)  # Ensure maxOutputTokens is within the valid range
        responses = model.generate_content(
            [prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True
        )
        # Collecting generated content
        content = ""
        for response in responses:
            if hasattr(response, 'text'):
                content += response.text
            else:
                print(f"Unexpected response format: {response}")

        print(content)
        return content
    
        

    except Exception as e:
        print(f"Error generating content: {e}")
        return ""

def generate_AI_Optimised_title(Brand,Gender,Material,Product_Type, Product_Category, Color, Print, Fit,  Sleeve_Type, Collar_type, Cuff_Style,Product_USPs, Pocket_Type, Logo_Details, Marketplace  ,category,generation_config, safety_settings):
    prompt = f"""
    Analyze the provided title for the {category} of the brand named {Brand}. You have an input which contains the attribute values for this Shirt. Apply your intelligence to make a title, adhering to the specified format by filling in the attribute values given in the input:
    
    You must generate the title in the following format:
        title = "<brand> <gender> <material> <color> <print> <fit> <product_type> <product_category> | <sleeve_type>, <collar_type>, <cuff_style>, <product_usps>, <pocket_type>, <logo_details> | <marketplace_name>"

    Now after you have generated the title, optimize the title by performing the tasks below. 

    Tasks:
    1. Remove undesirable words like "Polyester" or "Recycled Polyster Button and "Lyocell" from the entire title that people avoid while buying shirts. Ensure these words are completely excluded from any part of the title.
    2. Make the title more trendy by replacing keywords with words that have higher Monthly Search Volume (MSV). Use your intelligence to determine the more popular synonyms or trendy words for the given attributes.
    2. Optimize the title to be within 150 characters.

    

    Input:

        brand = {Brand}
        gender = {Gender}
        material = {Material}
        product_type = {Product_Type}
        product_category = {Product_Category}
        color = {Color}
        print_value = {Print}
        fit = {Fit}
        sleeve_type = {Sleeve_Type}
        collar_type = {Collar_type}
        cuff_style = {Cuff_Style}
        product_usps = {Product_USPs}
        pocket_type = {Pocket_Type}
        logo_details = {Logo_Details}
        marketplace_name = {Marketplace}
        
    Output Format:
         The title for the product is: ||| <optimized_title> |||
         
    Example:

        brand = Gant
        gender = Men
        material = Cotton
        product_type = Casual
        product_category = Shirt
        color = Black
        print_value = Checked
        fit = Regular Fit
        sleeve_type = Full Sleeve
        collar_type = Button-Down Collar
        cuff_style = Standard Cuff
        product_usps = Comfortable
        pocket_type = Single Pocket
        logo_details = Embroidered Logo
        marketplace_name = Iconic
        
        The title for the product is: ||| Gant Men Cotton Black Checked Regular Fit Casual Shirt | Full Sleeve, Button-Down Collar, Standard Cuff, Comfortable, Single Pocket, Embroidered Logo | Iconic |||
"""


    try:
        print("Generating content...")
        model = choose  # Ensure this is correctly initialized with your model
        # generation_config["maxOutputTokens"] = min(generation_config.get("maxOutputTokens", 2048), 2048)  # Ensure maxOutputTokens is within the valid range
        responses = model.generate_content(
            [prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True
        )
        # Collecting generated content
        content = ""
        for response in responses:
            if hasattr(response, 'text'):
                content += response.text
            else:
                print(f"Unexpected response format: {response}")

        print(content)
        return content
    
        

    except Exception as e:
        print(f"Error generating content: {e}")
        return ""


# Function to extract attribute values from analysis_result
def extract_attributes(result):
    attributes = {}
    lines = result.splitlines()
    print(f"Lines in result:\n{lines}")  # Debug statement to print all lines in the result
    for line in lines:
        if '|' in line:
            parts = [item.strip() for item in line.split('|')[1:3]]
            if len(parts) == 2:
                attribute, value = parts
                attributes[attribute] = value
                print(f"Extracted: Attribute - {attribute}, Value - {value}")  # Debug statement to print extracted attribute-value pair
            else:
                print(f"Skipping line: {line}. Expected 2 parts, got {len(parts)} parts.")
    print(f"Extracted attributes:\n{attributes}")  # Debug statement to print all extracted attributes
    return attributes

def generate_ai_titles(row):
    try:
        # Extract attributes from the row
        brand = row['Brand']
        gender = row['Gender']
        material = row['Material']
        product_type = row['Product Type']
        product_category = row['Product Category']
        color = row['Color']
        print_value = row['Print']
        fit = row['Fit']
        sleeve_type = row['Sleeve Type']
        collar_type = row['Collar Type']
        cuff_style = row['Cuff Style']
        product_usps = row['Product USPs']
        pocket_type = row['Pocket Type']
        logo_details = row['Logo Details']
        marketplace_name = row['Marketplace']
        
        # Check conditions for excluding attributes from the title
        exclude_values = ['', None, 'N/A','None','-']
        
        # Generate the title in the specified format
        title_parts = []
        
        if brand and brand.lower() not in exclude_values:
            title_parts.append(brand)
        if gender and gender.lower() not in exclude_values:
            title_parts.append(gender)
        if material and material.lower() not in exclude_values:
            title_parts.append(material)
        if color and color.lower() not in exclude_values:
            title_parts.append(color)
        if print_value and print_value.lower() not in exclude_values:
            title_parts.append(print_value)
        if fit and fit.lower() not in exclude_values:
            title_parts.append(fit)
        if product_type and product_type.lower() not in exclude_values:
            title_parts.append(product_type)
        if product_category and product_category.lower() not in exclude_values:
            title_parts.append(product_category)
            
        title_parts.append(" | ")
        
        attributes = [sleeve_type, collar_type, cuff_style, product_usps, pocket_type, logo_details]
        formatted_attributes = [attr for attr in attributes if attr and attr not in exclude_values]
        
        if formatted_attributes:
            title_parts.append(", ".join(formatted_attributes))
            
        title_parts.append("|")
        
        title_parts.append(marketplace_name)
        
        title = " ".join(title_parts)
        
        return title
    except Exception as e:
        print(f"Error generating title for row {row.name}: {e}")
        return ""

def extract_title(result):
    lines = result.splitlines()
    print(f"Lines in result:\n{lines}")  # Debug statement to print all lines in the result
    title = ""
    
    for line in lines:
        if 'The title for the product is:' in line:
            title_start = line.find('|||')
            title_end = line.rfind('|||')
            
            if title_start != -1 and title_end != -1 and title_start != title_end:
                title = line[title_start + 3:title_end].strip()
                print(f"Extracted title: {title}")  # Debug statement to print extracted title
                break
            else:
                print(f"Skipping line: {line}. Expected '|||' delimiters.")
    
    print(f"Final extracted title: {title}")  # Debug statement to print the final extracted title
    return title



# Function to calculate length of generated titles
def calculate_title_length(df):
    df['Length of title'] = df.apply(lambda row: len(generate_ai_titles(row)), axis=1)
    return df


# Function to calculate length of generated titles
def calculate_Optimised_title_length(df):
    df['Length of Optimised title'] = df.apply(lambda row: len(row['AI_Optimised_Title'] ), axis=1)
    return df



def main():
    
    csv_file = "Gant_Shirt.csv"
    df = pd.read_csv(csv_file)
    
    df5 = df
    
    
    filtered_df5 = filter_image_columns(df5)
    print(filtered_df5.columns)
    
    # Process additional image links if the column exists
    if 'images' in filtered_df5.columns:
        filtered_df5 = process_additional_image_links(filtered_df5, 'images')
        
     
    
    filtered_df5.drop(columns=['images'], inplace=True)
    
    filtered_df5 = rename_and_append_image_columns(filtered_df5)
    
    generation_config = {
    "temperature": 0.3
    }

    # Safety settings to adjust the level of filtering
    safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    }
    
    category = "Shirt"
    
    filtered_df5['analysis_result'] = filtered_df5.apply(
    lambda row: process_images_and_generate_content(
    [row['image_link'], row['image_url'], *row.filter(like='image_').dropna()],
    row['productName'],
    row['description'],
    row['productCategoryList'],
    row['tags'],
    row['more_info'],
    row['productCategoryLevel1'], 
    row['product_type'], 
    category,
    generation_config,
    safety_settings
    ), axis=1
    )
    
    filtered_df5_extracted = pd.DataFrame(filtered_df5['analysis_result'].apply(extract_attributes).tolist())

    
    # Generate AI titles for each row
    filtered_df5_extracted['AI Generated Title'] = filtered_df5_extracted.apply(generate_ai_titles, axis=1)
    
    filtered_df5_extracted=calculate_title_length(filtered_df5_extracted)
    
    
    filtered_df5_extracted['AI Optimised Title'] = filtered_df5_extracted.apply(
    lambda row: generate_AI_Optimised_title(
        row['Brand'],
        row['Gender'],
        row['Material'],
        row['Product Type'],
        row['Product Category'],
        row['Color'],
        row['Print'], 
        row['Fit'], 
        row['Sleeve Type'], 
        row['Collar Type'], 
        row['Cuff Style'], 
        row['Product USPs'], 
        row['Pocket Type'], 
        row['Logo Details'], 
        row['Marketplace'], 
        category,
        generation_config,
        safety_settings
    ), axis=1
    )
    
    
    filtered_df5_extracted_optimised_title = pd.DataFrame(filtered_df5_extracted['AI Optimised Title'].apply(extract_title).tolist())
    filtered_df5_extracted_optimised_title.columns = ['AI_Optimised_Title']
    
    
    filtered_df5_extracted_optimised_title=calculate_Optimised_title_length(filtered_df5_extracted_optimised_title)
    
    filtered_df5_extracted=pd.concat([filtered_df5_extracted.iloc[:, 2:],filtered_df5_extracted_optimised_title],axis=1)
    
    
    filtered_df5_Ex=pd.concat([filtered_df5,filtered_df5_extracted.iloc[:, 2:]],axis=1)


    output_csv = "output_Full_5.csv"
    filtered_df5_Ex.to_csv(output_csv, index=False)
    
    # Specify specific columns from df1 to concatenate
    specific_columns = ['productSku', 'pagePath','productName', 'description']
    
    filtered_df5_Ai=pd.concat([filtered_df5[specific_columns],filtered_df5_extracted.iloc[:, 2:]],axis=1)
    
    
    
    output_csv = "Sheet_present_Full_5.csv"
    filtered_df5_Ai.to_csv(output_csv, index=False)
    
    
    
if __name__ == "__main__":
    main()
