from reports_generator import ReportsGenerator
from fastapi import HTTPException

import pandas as pd


class RepGen:
    def __init__(self):
        self.summarizer = ReportsGenerator()

    def __call__(self, data):
        try:
            generated_report = self.summarizer(data)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
        return generated_report

    def handle_file(self, csv_data):
        generated_report = {}

        df = pd.read_csv(csv_data)
        try:
            assert ['id', 'original_text', 'groundtruth_labels'] == df.columns.to_list()
        except AssertionError:
            raise HTTPException(
                status_code=500,
                detail="Assertion faied on csv column names"
            )

        labels_list = df.groundtruth_labels.unique()

        try:
            for one_label in labels_list:
                mask_one_label = df.groundtruth_labels == one_label
                text_one_label = df[mask_one_label].original_text.tolist()
                generated_report[one_label] = self.summarizer(text_one_label)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

        return generated_report
