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

GOOGLE_API_KEY = "AIzaSyDH_IUXsgim26W9XQZ3rCxU1AsksbFevto"
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
| Style Type            |                 |
| Material              |                 |
| Product Type          |  {product_type} |
| Fit Type              |                 |
| Rise                  |                 |
| Length                |                 |
| Wash                  |                 |
| Color                 |                 |
| Product USPs          |                 |
| Marketplace           |     Iconic      |

Attributes to Look for:

Brand: Identify the brand of the {product_type}. Use the given {Brand} value.
Gender: Determine the intended demographic for the {product_type}, specifying whether it is targeted towards Men, Women, or Unisex preferences. This can be inferred from the provided text or visual cues from images.
Style Type: Identify the style type of the {product_type}, such as Boot-Cut, Flared, Mom-Fit, Relaxed, Slim, Straight, Tapered, Skinny, Boyfriend, Wide Leg, or Baggy. Use the provided text or images to infer the style type.
Material: Identify the fabric or material of the {product_type}, such as Denim, Cotton, Corduroy, Cotton Blend, or Gymindigo. This can be found in the provided text or packaging details in images. If the material cannot be inferred directly from the text description, analyze the images for any relevant symbols or indications. If still not found in the images, use overall knowledge from the text description and images to make an educated prediction. Always provide a predicted value rather than leaving the attribute blank or using placeholders like "none," "not specified," or "N/A."
Product Type: Confirm it is a {product_type}.
Fit Type: Identify the fit type of the {product_type}, such as Slim Fit, Regular Fit, Skinny Fit, Tapered Fit, Loose Fit, Relaxed Fit, or Straight Fit. If Slim Fit or Slim, specify Slim Fit, and for Regular Fit or Regular, specify Regular Fit. If Slim Fit or Slim, Give Slim Fit and For Regular Fit or Regular, Give Regular Fit.
Rise: Identify the rise of the {product_type}, such as Mid Waist, Low Waist, or High Waist.
Length: Identify the length of the {product_type}, such as Ankle Length, Full Length, or Capri. If the Neck Type cannot be inferred directly from the text description, analyze the images for any relevant symbols or indications. If still not found in the images, use overall knowledge from the text description and images to make an educated prediction. Always provide a predicted value rather than leaving the attribute blank or using placeholders like "None," "not specified," or "N/A."
Wash: Identify the wash of the {product_type}, such as Light Wash, Mid Wash, Clean, Heavy Wash, Heavily Faded, Non-Faded, Dark Rinse, or Faded Black, based on the provided text and images.
Color: Identify the color of the {product_type} from the description or images. Example values- Yellow, Black, Brown, Grey, Maroon, Blue, Green, Red, Pink, Beige, etc.
Product USPs: Identify any unique selling points or special features of the {product_type}, such as Wrinkle Free, Limited Edition, Iron Free, etc. If the Product USPs cannot be inferred directly from the text description, analyze the images for any relevant symbols or indications. If still not found in the images, use overall knowledge from the text description and images to make an educated prediction. Always provide a predicted value rather than leaving the attribute blank or using placeholders like "None," "not specified," or "N/A."
Marketplace: This is fixed as Iconic.
Please ensure that all attributes have predicted values based on available information from both textual descriptions and visual analysis of images. If specific details are not explicitly stated, use reasonable inference based on the overall context provided. This approach ensures comprehensive attribute prediction for the product.

Example:

Title: "Antony Morato Men Navy Blue Slim Fit Slash Knee Heavy Fade Stretchable Cotton Jeans"
Description: "Introducing the iconic Antony Morato Mens Blue Denim Jeans. Crafted with 98% cotton and 2% elastane, these jeans are designed for a slim fit and feature a slub stretch true blue denim.The perfect addition to any wardrobe, these iconic jeans from Antony Morato are sure to become a staple in your wardrobe. With a slim fit and stretchy fabric, you'll be comfortable and stylish all day long."
productCategoryList: Discount,Antony Morato Men Jeans,Summer Soiree Collection,Jeans for Men,Premium Clothing for Men,Gift For Loved Ones,Prepay 10,Tax 12%,Prepay,Men's/Women's Jeans,Sale,Antony Morato AM-PM,Antony Morato Clothing Collection for Men & Kids,Flat 50,Men's Bottomwear,Best Of the Season Collection,Antony Morato Men's Clothing Collection,All Iconic Editions
Tags: Antony Morato, Antony Morato AM-PM, Antony Morato Men, Antony Morato Men Jeans, ANTONY-MORATO-MEN-JEANS-SLIM-FIT, AW22CF, Blue, Description, Discount, Flat 50, Gift For Love Ones, Homepage 2, Jeans, Men, Men Bottomwear, Men Jeans, MMDT00242-FA750344-7010-W01524, Prepay 10, Slim Fit, Solid, Summer Soiree
more_info: "body_html": "<p>Introducing the iconic Antony Morato Mens Blue Denim Jeans. Crafted with 98% cotton and 2% elastane, these jeans are designed for a slim fit and feature a slub stretch true blue denim.<br><br>The perfect addition to any wardrobe, these iconic jeans from Antony Morato are sure to become a staple in your wardrobe. With a slim fit and stretchy fabric, you'll be comfortable and stylish all day long.</p>"




Output Format:
| Attribute             | Predicted Value |
|-----------------------|-----------------|
| Brand                 |   Antony Morato |
| Gender                |    Men          |
| Style Type            |   Strechable    |
| Material              |   Cotton        |
| Product Type          |  Jeans          |
| Fit Type              |   Slim Fit      |
| Rise                  |   Mid Rise      |
| Length                |  Full Length    |
| Wash                  |  Piece Dye      |
| Color                 |   Navy Blue     |
| Product USPs          |   Slash Knee    |
| Marketplace           |     Iconic      |
    """

    file_names = []
    for url in url_column:
        filename = download_image(url, "temp_image_1.jpg")
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



def generate_AI_Optimised_title( Brand,Gender,Style_Type, Material,Product_Type, Fit_Type, Rise,Length,Wash ,Color,Product_USPs,  Marketplace,category,generation_config, safety_settings):
    prompt = f"""
    Analyze the provided title for the {category} of the brand named {Brand}. You have an input which contains the attribute values for this Shirt. Apply your intelligence to make a title, adhering to the specified format by filling in the attribute values given in the input:
    
    You must generate the title in the following format:
        title = "<brand> <gender> <Style_type> <material> <product_type> <fit_type>| <Rise> , <Length>, <Wash> ,<color>, <product_usps> | <marketplace_name>"

    Now after you have generated the title, optimize the title by performing the tasks below. 

    Tasks:
    1. Remove undesirable words like "Polyester" or "Recycled Polyster Button and "Lyocell" from the entire title that people avoid while buying shirts. Ensure these words are completely excluded from any part of the title.
    2. Make the title more trendy by replacing keywords with words that have higher Monthly Search Volume (MSV). Use your intelligence to determine the more popular synonyms or trendy words for the given attributes.
    3. Change the color mentioned in the title to a color which is understood by layman (avoid hard-to-understand color terms).
    4. Optimize the title to be within 150 characters.

    

    Input:

        brand = {Brand}
        gender = {Gender}
        style_type = {Style_Type}
        material = {Material}
        product_type = {Product_Type}
        fit_type = {Fit_Type}
        Rise = {Rise}
        Length = {Length}
        Wash = {Wash}
        color = {Color}
        product_usps = {Product_USPs}
        marketplace_name = {Marketplace}
        
    Output Format:
         The title for the product is: ||| <optimized_title> |||
         
    Example:

        brand = Antony Morato
        gender = Men
        style_type = Strechable
        material = Cotton
        product_type = Jeans
        fit_type = Slim Fit
        Rise = Mid Rise
        Length = Full Length
        Wash = Piece Dye
        color = Navy Blue
        product_usps = Slash Knee
        marketplace_name = Iconic
        
        The title for the product is: ||| Antony Morato Strechable Cotton Jeans Slim Fit | Mid Rise, Full Length, Piece Dye , Navy Blue | Slash Knee | Iconic |||
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
        style_type = row['Style Type']
        material = row['Material']
        product_type = row['Product Type']
        fit_type = row['Fit Type']
        Rise= row['Rise']
        Length= row['Length']
        Wash = row['Wash']
        color = row['Color']
        product_usps = row['Product USPs']
        marketplace_name = row['Marketplace']
        
        # Check conditions for excluding attributes from the title
        exclude_values = ['', None, 'N/A','None','-','Not Specified']
        
        # Generate the title in the specified format
        title_parts = []
        
        if brand and brand.lower() not in exclude_values:
            title_parts.append(brand)
        if gender and gender.lower() not in exclude_values:
            title_parts.append(gender)
        if style_type and style_type.lower() not in exclude_values:
            title_parts.append(style_type)
        if material and material.lower() not in exclude_values:
            title_parts.append(material)
        if product_type and product_type.lower() not in exclude_values:
            title_parts.append(product_type)
        if fit_type and fit_type.lower() not in exclude_values:
            title_parts.append(fit_type)

            
        title_parts.append(" | ")
        
        attributes = [Rise, Length, Wash, color]
        formatted_attributes = [attr for attr in attributes if attr and attr not in exclude_values]
        
        if formatted_attributes:
            title_parts.append(", ".join(formatted_attributes))
            
        title_parts.append("|")
        
        if product_usps and product_usps.lower() not in exclude_values:
            title_parts.append(product_usps)
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
    
    csv_file = "dataframe1.csv"
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
    
    category = "Jeans"
    
    filtered_df5['analysis_result'] = filtered_df5.apply(
    lambda row: process_images_and_generate_content(
    [row['image_link'], row['image_url'], *row.filter(like='image_').dropna()],
    row['productName'],
    row['description'],
    row['productCategoryList'],
    row['tags'],
    row['more_info'],
    row['vendor'], 
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
        row['Style Type'],
        row['Material'],
        row['Product Type'],
        row['Fit Type'], 
        row['Rise'], 
        row['Length'], 
        row['Wash'], 
        row['Color'],
        row['Product USPs'], 
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
    
    
    filtered_df5_Ex=pd.concat([filtered_df5,filtered_df5_extracted],axis=1)


    output_csv = "output_Full_Jeans_1.csv"
    filtered_df5_Ex.to_csv(output_csv, index=False)
    
    # Specify specific columns from df1 to concatenate
    specific_columns = ['productSku', 'pagePath','productName', 'description']
    
    filtered_df5_Ai=pd.concat([filtered_df5[specific_columns],filtered_df5_extracted],axis=1)
    
    
    
    output_csv = "Sheet_present_Jeans_1.csv"
    filtered_df5_Ai.to_csv(output_csv, index=False)
    
    
    
if __name__ == "__main__":
    main()
