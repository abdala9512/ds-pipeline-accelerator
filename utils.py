"""Submission related functions"""



class Submission:

    def __init__(self, submission_template, predictor):
        self.template = submission_template
        self.predictor = predictor
        pass


    def create_submission(self, predictions, model_name):
    """Create a csv to Kaggle submission

    Args:
        predictions ([type]): Array of predictions
        model_name ([type]): Model name in the csv file
        model_submission (pandas.DataFrame): submission template
        predictor_name ([type]): predictor name in the submission fiel
    """
    
    self.template[self.predictor] = predictions
    self.template.to_csv("submissions/" + model_name + "_prediction.csv", index = False)