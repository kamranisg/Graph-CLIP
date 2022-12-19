import pandas as pd
import pickle
import json


def extract_reports_425(data_425,df):

    report_425 = list(data_425)
    new_reports_425=[]

    for idx,report in enumerate(report_425):
        x=report.split("/")
        pid = x[1]
        sid = x[2].split(".")[0]
        row = (df[(df.patient_id == int(pid[1:]))  & (df.study_id == int(sid[1:]))])
        if not row.empty:
            new_reports_425.append(report)
    
    with open("Frontal_Graph_425.pkl",'wb') as f:
        pickle.dump(new_reports_425,f)

def extract_reports_3000(data,df):

    report_3000 = list(data)[:3000]
    print(len(report_3000))
    new_reports_3000=[]

    for idx,report in enumerate(report_3000):
        x=report.split("/")
        pid = x[1]
        sid = x[2].split(".")[0]
        row = (df[(df.patient_id == int(pid[1:]))  & (df.study_id == int(sid[1:]))])
        if not row.empty:
            new_reports_3000.append(report)
    
    print(len(new_reports_3000))
    with open("Frontal_Graph_3000.pkl",'wb') as f:
        pickle.dump(new_reports_3000,f)

def main():

    df=pd.read_csv("/vol/space/projects/MIMIC-CXR/mimic-cxr_ap-pa_dataset/combined_frontal_list.csv")
    file="/u/home/mohammed/Graphormer/examples/customized_dataset/MIMIC-CXR_graphs.json"
    file_425="/u/home/mohammed/Graphormer/examples/customized_dataset/train.json"
    with open(file) as f:
         data = json.load(f)
    
    with open(file_425) as f:
         data_425 = json.load(f)

    extract_reports_3000(data,df)


if __name__ == '__main__':
    main()