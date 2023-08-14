export const API_URLS = {
    WORDS: `https://cklvhhyl66.execute-api.us-east-1.amazonaws.com/`,
    ISOLATED_SIGNS: `https://isolatedsigns.s3.amazonaws.com/`,
    SENTENCE_SIGNS: `https://sentencesigns.s3.amazonaws.com/`
}

export const AWS_CONFIG = {
    REGION: `us-east-1`,
    IDENTITY_POOL_ID: `us-east-1:b5013574-2741-4e18-97be-9395b5929162`,
    LAMBDA_SAGEMAKER_INOKER: `sagemaker-invoker`
}

export const RECORDING = {
    FPS: 1000 / 30,
    PREPARATION_TIME: 3, //seconds
    RECORDING_TIME: 2500, //ms
}
