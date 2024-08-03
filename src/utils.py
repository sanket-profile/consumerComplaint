import re


def change_product(text):
  if text == "Other financial service":
    return "Credit Reporting"
  else:
    return text
  
def merge_values(x,fromThis,to):
  if x in fromThis:
    return to
  else:
    return x
  
def preprocess_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\d", " ", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("\[.*?\]", "", text)
    text = re.sub(r"""[.,/""/'':-]""", '', text) #Removes special character from the text
    text = re.sub(r'xxxx', '', text) # Removes xxxx pattern from the text
    text = re.sub(r'[\{\}\$]', '', text)  # Remove curly braces and dollar signs
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.lower()
    return text