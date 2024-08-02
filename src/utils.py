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
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\d", " ", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("\[.*?\]", "", text)
    return text