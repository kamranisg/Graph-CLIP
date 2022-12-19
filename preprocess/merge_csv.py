import pandas as pd

def main():

    df1=pd.read_csv("/vol/space/projects/MIMIC-CXR/mimic-cxr_ap-pa_dataset/train/image_list.csv")
    df2=pd.read_csv("/vol/space/projects/MIMIC-CXR/mimic-cxr_ap-pa_dataset/validation/image_list.csv")
    df3=pd.read_csv("/vol/space/projects/MIMIC-CXR/mimic-cxr_ap-pa_dataset/test/image_list.csv")
    print(len(df1),len(df2),len(df3))
    final_df=pd.concat([df1,df2,df3]).reset_index(drop=True)
    final_df=final_df.reset_index(drop=True)
    print(len(final_df))
    #final_df.to_csv("/vol/space/projects/MIMIC-CXR/mimic-cxr_ap-pa_dataset/combined_frontal_list.csv",index=False)


if __name__ == '__main__':
    main()