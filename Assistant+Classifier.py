
# coding: utf-8

# # classifier

# In[1]:

get_ipython().system('pip install pandas_ml')
get_ipython().system('pip install -I watson-developer-cloud==1.3.3')


# In[5]:

import json
import sys
import codecs
import unicodecsv as csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas_ml
from pandas_ml import ConfusionMatrix
from watson_developer_cloud import WatsonApiException
from watson_developer_cloud import AssistantV1


# In[ ]:

# params
convParmsFile = '/Users/jrnash/sretan.json'
parms = ''
with open(convParmsFile) as parmFile:
    parms = json.load(parmFile)

url=parms['url']
user=parms['user']
password=parms['password']
workspace_id=parms['workspace_id']
test_csv_file=parms['test_csv_file']
results_csv_file=parms['results_csv_file']
confmatrix_csv_file=parms['confmatrix_csv_file']

assistant = AssistantV1(
  username=user,
  password=password,
  version='2018-02-16')

assistant.set_http_config({'timeout': 100})


# In[ ]:

#test

def get_res(assist_instance,workspaceID,string):
    
    context = {}    
    string = string.replace("\n","")
    res = assist_instance.message(
        workspace_id=workspaceID,
        input={'text':string})
    classes=res
    return classes

    # context object not necessary for call
    # alternate_intents=True in call for other intents

    
# batches from csv of text

def batch(assist_instance,workspaceID,csvfile):
    test_class=[]
    assist_predict_classes=[]
    assist_predict_confidence=[]
    text=[]
    j=0
    print ('csv: ', csvfile)
    
    
    with open(csvfile,"rb") as csvfile:
        csvReader = csv.reader(csvfile, encoding="utf-8-sig")
        
        for row in csvReader:
            if len(row) > 2:
                elements = row[0:len(row)-1]
                utter = ",".join(elements)
                test_class.append(row[len(row)-1])
            else:
                utter = row[0]
                test_class.append(row[1])
            utter = utter.replace('\r', ' ')
            print (' test: ', j, 'testing row ', utter)
            
            
            assist_response = get_res(assist_instance,workspaceID,utter)

            if assist_response['intents']:
                assist_predict_classes.append(assist_response['intents'][0]['intent'])
                assist_predict_confidence.append(assist_response['intents'][0]['confidence'])
            else:
                assist_predict_classes.append('')
                assist_predict_confidence.append(0)
            text.append(utter)
            j = j+1
            # counts on processed
            if(j%250 == 0):
                print("")
                print('processed ',j, ' records')
            
            if(j%10 == 0):
                sys.stdout.write('.'),
        print('finished!! ', j, '  records')
    return test_class, assist_predict_classes, assist_predict_confidence, text

def conf_matrix(conf_matrix):
    plt.figure()
    plt.imshow(conf_matrix)
    plt.show()

    def matrixforcsv(conf_matrix,labels,csvfile):
        with open(csvfile, 'wb') as csvfile:
            csvWriter = csv.writer(csvfile)
            row=list(labels)
            row.insert(0,"")
            csvWriter.writerow(row)
            for i in range(conf_matrix.shape[0]):
                row=list(conf_matrix[j])
                row.insert(0, labels[j])
                csvWriter.writerow(row)
test = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum"
result = get_res(assistant, workspace_id, test)
print(json.dumps(result, indent=2))


# In[ ]:

#remove alternate_intents=True from get_res function

test_class, assist_predict_classes, assist_predict_confidence, text=batch(assistant,workspace_id,test_csv_file)


# In[ ]:

# print

csvfileOut = results_csv_file
csvWriter = codecs.open(csvfileOut, 'w', encoding="utf-8-sig")

outrow = ['Comments','True Class','Watson Predicted Class','Confidence']
csvWriter.write("Comments"+","+"True Class"+","+"Watson Predicted Class"+","+"Confidence")
csvWriter.write('\n')


for j in range(len(text)):
    t = text[j]
    txtx = text[j]
    if txtx[:1] != '"' or txtx.strip()[-1] != '"':
        txtx = "\"" + txtx + "\""
        
    csvWriter.write(txtx+","+test_class[j]+","+str(assist_predict_classes[j])+","+str(assist_predict_confidence[j]))
    csvWriter.write("\n")
csvWriter.close()


# In[ ]:



