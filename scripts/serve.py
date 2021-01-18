# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import logging
import json
import io
import os
from fastai.vision.all import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
EXPORT_MODEL_NAME = 'model.pth'


# loads the model into memory from disk and returns it
def model_fn(model_dir):
    logger.info('model_fn')
    path_model = Path(model_dir)
    logger.debug(f'Loading model from path: {str(path_model/EXPORT_MODEL_NAME)}')
    defaults.device = torch.device('cpu')
    learn = load_learner(path_model/EXPORT_MODEL_NAME, cpu=True)
    logger.info('model loaded successfully')
    return learn


# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        logger.info(f'Downloading image from URL: {url}')
        img_content = requests.get(url).content
        logger.info(f'Returning image bytes')
        return io.BytesIO(img_content).read()
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

    
# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_class,predict_idx,predict_values = model.predict(input_object)
    logger.info("--- Inference time: %s seconds ---" % (time.time() - start_time))
    predict_class=os.path.basename(predict_class)
    logger.info(f'Predicted class is {str(predict_class)}')
    logger.info(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    return { "class": str(predict_class),
        "confidence": predict_values[predict_idx.item()].item() }


# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: 
        logger.debug(f'Returning response {json.dumps(prediction)}')
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))