# python integrate_df in_directory out_directory
import sys
import pandas as pd
import os
from typing import List


class CO2DataFrameIntegrator:
    def __init__(
            self,
            in_directory: str,
            out_directory: str
    ):
        super().__init__()
        self.in_directory = in_directory
        self.out_directory = out_directory

    def enumerate_xls_files(self):
        """

        Returns: [Excel Files Path String]
        """
        xls_files = []

        # self.dataset_dir 안의 모든 파일 중에서
        for file_name in os.listdir(self.in_directory):
            file_path = os.path.join(self.in_directory, file_name)

            # 확장자가 csv인 파일을 csv_files에 모음
            if file_path.endswith('.xlsx'):
                xls_files.append(file_path)

        # csv_files를 알파벳 순서대로 정렬
        xls_files.sort()

        # csv 파일 경로 리스트 반환
        return xls_files

    # TODO: implement later
    def transform(self) -> None:
        xls_files = self.enumerate_xls_files()

        df = None
        for i, xls_path in enumerate(xls_files):
            print(f"[{i+1}/{len(xls_files)}")
            sub_df1 = pd.read_excel(xls_path, sheet_name=0, header=[0, 1], index_col=0)
            sub_df2 = pd.read_excel(xls_path, sheet_name=1, header=[0, 1], index_col=0)
            sub_df3 = pd.read_excel(xls_path, sheet_name=2, header=[0, 1], index_col=0)

            sub_df1 = sub_df1[~sub_df1.index.duplicated(keep='first')]
            sub_df2 = sub_df2[~sub_df2.index.duplicated(keep='first')]
            sub_df3 = sub_df3[~sub_df3.index.duplicated(keep='first')]

            sub_df = sub_df1.join(sub_df2, how='inner')
            sub_df = sub_df.join(sub_df3, how='inner')

            sub_df.drop(['AVG', 'MAX', 'MIN', 'STD'], inplace=True)
            #df.sort_index(inplace=True)

            #target_path = xls_path + '.pkl'
            #df.to_pickle(target_path)
            if df is None:
                df = sub_df
            else:
                df = pd.concat([df, sub_df])

        # sort df by column ??
        df.sort_index(inplace=True)

        out_file = os.path.join(self.out_directory, "integrated.pkl")
        df.to_pickle(out_file)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Argument is lack')

    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    integrator = CO2DataFrameIntegrator(in_directory, out_directory)
    integrator.transform()
