from ipykernel import kernelapp as app
from requests import get
from bs4 import BeautifulSoup
from sklearn.externals import joblib
from lime.lime_text import LimeTextExplainer


def run_model(link):
    print('\n\n')
    #print('Please enter your link to be classified')
    #link = input()
    title, keywords, description = '', '', ''
    try:
        response = get(link)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        if soup.title:
            title = soup.title.string
        for tag in soup.find_all("meta"):
            if tag.get("name", None) == "keywords":
                keywords = tag.get("content", None)
            if tag.get("name", None) == "description":
                description = tag.get("content", None)
        text = title+keywords+description
    except ConnectionError as e:
        text = "No Response"
    
    clf = joblib.load('LR.joblib')
    if clf.predict([text])[0] == 1:
        result = '\nYour Link is Classified as:\n<==========Related==========>\n'
        print(result)
    else:
        result = '\nYour Link is Classified as:\n<==========Not_related==========>\n'
        print(result)
    #print(clf.predict_proba([text]))
    return (result, text)


def lime_explanation(link):

    result, text = run_model(link)
    explainer = LimeTextExplainer(class_names=[1, 0])
    clf = joblib.load('LR.joblib')
    exp = explainer.explain_instance(text, clf.predict_proba, labels=[0,1], num_features=10)
    fig = exp.as_pyplot_figure()
    fig.savefig('image.jpg')
    result = result + str('Predicted class ={}'.format(clf.predict([text]).reshape(1,-1)[0,0]))
    result= result + str('\nExplanation for class [0]:\n{}'.format('\n'.join(map(str, exp.as_list(label=0)))))
    result= result + str('\n\nExplanation for class [1]:\n{}'.format('\n'.join(map(str, exp.as_list(label=1)))))
    result = result.split('\n')
    return(result)
