import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
from tkinter import Tk, Text, Scrollbar, Entry, Button, INSERT, RIGHT, Y, LEFT, BOTH, StringVar
from tkinter import *
warnings.filterwarnings("ignore", category=DeprecationWarning)

global Name     # Declare Name as a global variable inside the function
global disease_input
global num_days
global symptom
disease_input = None
num_days = None
symptoms_present = []
symptoms_exp = []
present_disease = None

def chatexit():
    root.destroy()

# def display_symptoms():

#     # Display the symptoms in a dialogue box
#     messagebox.showinfo("All Symptoms", symptoms_text)

def display_symptoms():
    global symptoms_dict, text_widget

    # Create the root window
    root2 = Tk()
    root2.title("Symptoms")
    root2.geometry("400x400")

    # Create a Text widget inside the root window
    text_widget = Text(root2)
    text_widget.pack(expand=True, fill=BOTH)

    # Add a scroll bar to the Text widget
    scroll_bar = Scrollbar(root2)
    scroll_bar.pack(side=RIGHT, fill=Y)
    text_widget.pack(side=LEFT, fill=BOTH)

    # Configure the Text widget to use the scroll bar
    scroll_bar.config(command=text_widget.yview)
    text_widget.config(yscrollcommand=scroll_bar.set)

    # Initial display of all symptoms
    all_symptoms = list(symptoms_dict.keys())
    symptoms_text = "\n".join(all_symptoms)
    text_widget.insert(INSERT, symptoms_text)

    # Run the main loop for the root window
    root2.mainloop()
    
root = Tk()
root.config(bd=5)

# Create the menu bar
menu = Menu(root)
root.config(menu=menu)

# Create File menu
file_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Clear")
file_menu.add_command(label="Exit", command=chatexit)

# Create Symptoms menu
symptoms_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Symptoms", menu=symptoms_menu)
symptoms_menu.add_command(label="Display Symptoms", command= display_symptoms)

text_frame = Frame(root, bg="white")
text_frame.pack(fill="both", expand=True)

text_box = Text(text_frame, yscrollcommand=None, state=DISABLED,
                         bd=1, padx=6, pady=6, spacing3=8, wrap=WORD, bg=None, font="Verdana 10", relief=GROOVE,
                         width=10, height=1)
text_box.pack(expand=True, fill=BOTH)

training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y


reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
print (scores.mean())


model=SVC()
model.fit(x_train,y_train)
print("for svm: ")
print(model.score(x_test,y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t",end="->")
    name=input("")
    print("Hello, ",name)

def get_name():
    string=("--------------HealthCare ChatBot--------------\n" +
           "How May I call you?")
    return string

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

syms = None
symptoms_given = []
def recurse(node, depth, tree, feature_names, disease_input, num_days):
        global syms
        global present_disease
        global symptoms_given
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        global symptoms_present
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            print("if Triggered lv 1")
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
                print("if Triggered lv 2")
            else:
                val = 0
                print("else Triggered lv 1")
            if  val <= threshold:
                print("if Triggered lv 2 second one")
                recurse(tree_.children_left[node], depth + 1, clf, cols, disease_input, num_days)
            else:
                print("else Triggered lv 2 second one")
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1, clf, cols, disease_input, num_days)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            syms = symptoms_given[0]
            message_to_show = ("Bot: Correct ? \n")
            text_box.configure(state=NORMAL)
            text_box.insert(END, message_to_show)
            text_box.configure(state=DISABLED)
            text_box.see(END)
            red_cols = reduced_data.columns 
            print(present_disease)
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

idx = 0

def askMoreSymptoms():
    global symptoms_present
    global present_disease
    global symptoms_exp
    global symptoms_given
    global idx
    syms = symptoms_given[idx]
    message_to_show = ("Bot: Are You experiencing any " + syms + "\n")
    text_box.configure(state=NORMAL)
    text_box.insert(END, message_to_show)
    text_box.configure(state=DISABLED)
    text_box.see(END)
    idx+=1
    print(idx)
    print
    if idx == len(symptoms_given)-1:
        calcluateWhenEnough()

def calcluateWhenEnough():
    global num_days
    second_prediction=sec_predict(symptoms_exp)
    # print(second_prediction)
    num_days = int(num_days)
    calc_condition(symptoms_exp,num_days)
    if(present_disease[0]==second_prediction[0]):
        print("works")

        # readn(f"You may have {present_disease[0]}")
        # readn(f"{description_list[present_disease[0]]}")

    else:
        message_to_show = ("Bot: You may have " + present_disease[0] 
                           + " or " + second_prediction[0] + "\n\n"
                           + present_disease[0] + "\n"
                           + description_list[present_disease[0]] + "\n\n"
                           + second_prediction[0]
                           + description_list[second_prediction[0]] + "\n")
        text_box.configure(state=NORMAL)
        text_box.insert(END, message_to_show)
        text_box.configure(state=DISABLED)
        text_box.see(END)
        print("You may have ", present_disease[0], "or ", second_prediction[0])
        print(description_list[present_disease[0]])
        print(description_list[second_prediction[0]])

    # print(description_list[present_disease[0]])
        precution_list=precautionDictionary[present_disease[0]]
        message_to_show = ("Bot: Please Take the following measures \n")
        text_box.configure(state=NORMAL)
        text_box.insert(END, message_to_show)
        text_box.configure(state=DISABLED)
        text_box.see(END)
        for  i,j in enumerate(precution_list):
            x = i+1
            message_to_show = (str(x) + ")" + j + "\n")
            text_box.configure(state=NORMAL)
            text_box.insert(END, message_to_show)
            text_box.configure(state=DISABLED)
            text_box.see(END)
            print(i+1,")",j)

    # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
    # print("confidence level is " + str(confidence_level))


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def clearDisease():
    global disease_input
    disease_input = None

def clearDays():
    global num_days
    num_days = None


def tree_to_code(tree, feature_names, disease_input):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")

    # message_to_show = "Enter the symptom you are experiencing "
    # text_box.configure(state=NORMAL)
    # text_box.insert(END, message_to_show)
    # text_box.configure(state=DISABLED)
    # text_box.see(END)

    if disease_input:
        print("Nice ", end="->" + disease_input)

        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            for num,it in enumerate(cnf_dis):

                text_box.configure(state=NORMAL)
                text_box.insert(END, f"Bot: You are experiencing {it}.\n")
                text_box.configure(state=DISABLED)
                text_box.see(END)

            if num!=0:
                text_box.configure(state=NORMAL)
                text_box.insert(END, f"Bot: You are experiencing {num}.")
                text_box.configure(state=DISABLED)
                text_box.see(END)

                conf_inp = int(entry_field.get())
                disease_input=cnf_dis[conf_inp]
                tree_to_code(tree, feature_names, disease_input)
            while True:
                try:
                    message_to_show = "Bot: Okay. For how many days ? : \n"
                    text_box.configure(state=NORMAL)
                    text_box.insert(END, message_to_show)
                    text_box.configure(state=DISABLED)
                    text_box.see(END)
                    break
                except:
                    print("Bot: Enter valid input.")
            else:
                conf_inp=0

                # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
                # conf_inp = input("")
                # if(conf_inp=="yes"):
                #     continue
        else:
            text_box.configure(state=NORMAL)
            text_box.insert(END, "Bot: Enter valid symptom.\n")
            text_box.configure(state=DISABLED)
            text_box.see(END)
            # disease_input = None
            disease_input = "New thing"
            print("Disease Input " + disease_input)
            disease_input = None
            print(disease_input)
            clearDisease()
    else:
        text_box.configure(state=NORMAL)
        text_box.insert(END, "Bot: Please enter a symptom.\n")
        text_box.configure(state=DISABLED)
        text_box.see(END)

def num_days_assign(clf, cols, num_days, disease_input, ):
    try:
        message_to_show = ("Bot: You have been experiencing " + disease_input +
                        " for " + str(num_days) + " days \n")
        num_days = int(num_days)
        text_box.configure(state=NORMAL)
        text_box.insert(END, message_to_show)
        text_box.configure(state=DISABLED)
        text_box.see(END)
        recurse(0, 1, clf, cols, disease_input, num_days)
    except ValueError:
        message_to_show = ("Bot: Please Enter a number. \n")
        text_box.configure(state=NORMAL)
        text_box.insert(END, message_to_show)
        text_box.configure(state=DISABLED)
        text_box.see(END)
        clearDays()


getSeverityDict()
getDescription()
getprecautionDict()
# getInfo()
# tree_to_code(clf,cols)
print("----------------------------------------------------------------------------------------")

Name = None  # Declare Name as a global variable
symptom = None

def send_message_insert():
    global Name     # Declare Name as a global variable inside the function
    global disease_input
    global num_days
    global symptom
    user_input = entry_field.get()
    if user_input:
        if not Name:  # Check if Name is not set yet
            Name = user_input
            pr1 = "Human : " + user_input + "\n"
            text_box.configure(state=NORMAL)
            text_box.insert(END, pr1)
            text_box.configure(state=DISABLED)
            text_box.see(END)
            entry_field.delete(0, END)

            message_to_show = "Hello " + Name + "! Let's start by asking you some questions about your symptoms." + "\n"
            text_box.configure(state=NORMAL)
            text_box.insert(END, message_to_show)
            text_box.configure(state=DISABLED)
            text_box.see(END)
            entry_field.delete(0, END)
            # tree_to_code(clf, cols)
            
        elif (disease_input == None):
            disease_input =  user_input
            pr1 = "Human : " + user_input + "\n"
            text_box.configure(state=NORMAL)
            text_box.insert(END, pr1)
            text_box.configure(state=DISABLED)
            text_box.see(END)
            tree_to_code(clf, cols, disease_input)
            entry_field.delete(0, END)
        
        elif (num_days == None):
            print(disease_input)
            pr1 = "Human : " + user_input + "\n"
            text_box.configure(state=NORMAL)
            text_box.insert(END, pr1)
            text_box.configure(state=DISABLED)
            text_box.see(END)
            num_days =  user_input
            print(type(num_days))
            num_days_assign(clf, cols, num_days, disease_input, )
            entry_field.delete(0, END)

        else:
            symptom = user_input
            pr1 = "Human : " + user_input + "\n"
            text_box.configure(state=NORMAL)
            text_box.insert(END, pr1)
            text_box.configure(state=DISABLED)
            text_box.see(END)
            global syms
            if(user_input=="yes"):
                symptoms_exp.append(syms)
            askMoreSymptoms()
            entry_field.delete(0, END)

    else:
        text_box.configure(state=NORMAL)
        text_box.insert(END, "Error", "Please enter a message.")
        text_box.configure(state=DISABLED)
        text_box.see(END)

text_box_scrollbar = Scrollbar(text_frame)
text_box_scrollbar.pack(side=RIGHT, fill=Y)
text_box_scrollbar.config(command=text_box.yview)

# Create a frame for the user entry field
entry_frame = Frame(root, bd=1)
entry_frame.pack(side=LEFT, fill=BOTH, expand=True)

# Create a user entry field
entry_field = Entry(entry_frame, bd=1, justify=LEFT)
entry_field.pack(fill=X, padx=6, pady=6, ipady=3)

# Create a frame for the send and emoji buttons
send_button_frame = Frame(root, bd=0)
send_button_frame.pack(fill=BOTH)

# Create a send button & Emoji
send_button = Button(send_button_frame, text="Send", command=send_message_insert)
send_button.pack(side=LEFT)
entry_field.bind("<Return>", lambda event: send_message_insert())
emoji_button = Button(send_button_frame, text="Emoji")
emoji_button.pack(side=LEFT)

ob = get_name()
pr = "bot: " + ob + "\n"
text_box.configure(state="normal")
text_box.insert("end", pr)
text_box.configure(state="disabled")
text_box.see("end")
entry_field.delete(0, "end")

root.geometry("400x400")
root.mainloop()