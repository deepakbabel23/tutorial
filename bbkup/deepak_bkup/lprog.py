import pandas as pd
import numpy as np

df=pd.read_excel('//home//deepak//Downloads//TestCases.xlsx')
# df1=pd.read_excel('D:\Listany\TestCases.xlsx', usecols='TestCaseTitle')
# # print (df1)
fg=df.values.tolist()
# print(fg)
numOfRows = df.shape[0]
prevTestCase=[]
# testcase=[]
testcase ={'Title':0, 'Number':1,'Steps':2, 'Automate':3}
step=''
columns=list(df)
# print(columns)
# for x in range(numOfRows) :
list=[]
templist=[]
list1=[]
i=0
j=0
#
# # k=0
# # print("aoarf", df.iloc[0,0])
# for i in df.index:
#     # if i==1 :
#     #     exit()
#     print(i)
#     # for column in df:
#     #     if column=='TestCaseNumber':
#     #         exit()
#         #     columnSeriesObj = df[column]
#         #     value=columnSeriesObj.values
#         # df=df.replace(to_replace=['nan'], value=None)
#         df=df.fillna(0,inplace=True)
#         # df = df.replace({pd.np.nan: None})
#         print("after converting:", df)
#         # print(value)
#         # print(len(value))
#
#         list=df.iloc[i]
#         if list[0]==0:
#             list1=list[3]
#
#     # for k in range(len(value)):
#         #     print("The value of k", k)
#         #     print(value[k])
#         #     if (value[k]!=0):
#         #         val=value[k]
#         #         list.append(val)
#         #     else:
#         #         list1.append(value[k])
#         # print("the valuslist", list)
#         # print(list1
#
#
#
#       )# elif(column=='TestCaseTitle') :
#             #     testcase['Title']=val
#             #     print(testcase)
#             # elif (column == 'TestCaseNumber'):
#             #     testcase.setNumber(val)
#             # elif (column == 'Steps'):
#             #     testcase.setSteps(val)
#             # elif (column == 'Automate'):
#             #     testcase.setAutomate(val)
#             # print(testcase)
#         #     if (testcase[0] != None and len(testcase[0]) > 0):
#         #           testcase.getSteps().append(step)
#         #           if (prevTestCase != None and prevTestCase.getAutomate() != None and prevTestCase.getAutomate().equalsIgnoreCase("Y")):
#         #              testcase.append(prevTestCase)
#         #           prevTestCase = testcase
#         #     else:
#         #          prevTestCase.getSteps().append(step)
#         # if (prevTestCase != None and prevTestCase.getAutomate() != None and prevTestCase.getAutomate().equalsIgnoreCase("Y")):
#         #
#         #         testcase.add(prevTestCase)
#
# print(testcase)
#
#
#
#
#
#
# # print(testcase)
