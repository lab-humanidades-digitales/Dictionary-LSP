import { default as SDK } from 'aws-sdk';
import { API_URLS, AWS_CONFIG } from 'constants'

const AWS = SDK;

AWS.init = async () => {
    SDK.config.region = AWS_CONFIG.REGION;

    SDK.config.update({
        region: AWS_CONFIG.REGION,
        credentials: new SDK.CognitoIdentityCredentials({
            IdentityPoolId: AWS_CONFIG.IDENTITY_POOL_ID,
        }),
    });

    SDK.config.credentials.get((err) => {
        if (err) {
            console.error('Error retrieving AWS credentials:', err);
        }
    });
}

AWS.init();

export default AWS
