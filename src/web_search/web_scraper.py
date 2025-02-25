from googlesearch import search
from bs4 import BeautifulSoup
import requests


def taker(question):
    '''Search top 10 websites' URLs'''
    return list(search(question, num_results=10, lang='en'))


def sorter(websites):
    '''Sort first accessible website from top 10 websites'''
    for url in websites:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response  # Return the successful response directly
            else:
                print(f"Failed to access {url}: Status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error accessing {url}: {e}")
    return None  # If no websites were accessible


def Answer(response):
    '''Print sorted website's paragraphs'''
    if response is None:
        print("No accessible websites found.")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    limits = int(input("How many sentences do you want to print?: "))

    p_tag = soup.find_all('p')
    sentences = []

    for paragraph in p_tag:
        p_text = paragraph.get_text(strip=True)
        sentences.extend(p_text.split('.'))

    limits = min(limits, len(sentences))  # Prevent index error

    print("\n\n")
    for i in range(limits):
        print(sentences[i].strip(), end='')
        if i != limits - 1:
            print(". ", end='\n')
    print("\n\n")


# Main Execution
que = input("Enter your question: ")
websites = taker(que)
response = sorter(websites)
Answer(response)
