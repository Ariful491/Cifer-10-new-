


import numpy as np
import pandas as pd
import os
# project modules
from .. import config
from . import preprocess, my_model
# loading model
model = my_model.read_model()
label = [
    "airplane" ,"automobile" ,"bird" ,"cat" ,"deer" ,"dog" ,"frog" ,"horse" ,"ship" ,"truck"
]

# loading test data 
result = []
for part in  range(0,6):
    x_test = preprocess.get_test_data_by_part(part)

    # predicting results
    print("predicting results")
    predictions = model.predict(x_test,
                                batch_size=config.batch_size,
                                verbose = 2)

    lab_pred = np.argmax(predictions,axis=1)
    
    print(lab_pred)

    result += lab_pred.tolist()
    # print(result)


# submitting result into submission file
result = [label[i] for i in result]
# print(result)

submit_df= pd.DataFrame({"id": range(1, config.nb_test_samples +1),
                    "label":result})


submit_df.to_csv(os.path.join(config.submission_path(),"baseline_submission.csv"),
                  header=True, index=False,)

print("Boss Submission file Saved Successfully !")

