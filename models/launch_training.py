import os
import sys
import time
from datetime import datetime
import json

import openai
from openai import OpenAI
from validate_json import validate_json


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def launch_training(data_path: str) -> None:
    validate_json(data_path)
    
    # TODO: figure out how to specify file name in the new API
    # file_name = os.path.basename(data_path)

    # upload file
    with open(data_path, "rb") as f:
        output = client.files.create(
            file=f,
            purpose="fine-tune",
        )

    # TODO: save output to json file
    # client_output_metadata = 'openai_training_metadata_' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.json'
    # with open(client_output_metadata, 'w') as f:
    #     json.dump(output, f)

    print("File uploaded. Launching training job with information : {}".format(output))

    # launch training
    while True:
        try:
            job_output = client.fine_tuning.jobs.create(
                training_file=output.id,
                model="gpt-3.5-turbo-1106",
                suffix="hiep",
            )
            print('Job output: {}'.format(job_output))
            break
        except openai.BadRequestError:
            print("Waiting for file to be ready...")
            time.sleep(60)
    print(f"Training job {output.id} launched. You will be emailed when it's complete.")

if __name__ == "__main__":
    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        raise ValueError(f"Path {data_path} does not exist")

    launch_training(data_path)
