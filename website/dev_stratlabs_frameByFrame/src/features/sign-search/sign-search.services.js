/* eslint-disable no-debugger */
import API from "utils/axios-config";
import AWS from "utils/aws-config";
import { AWS_CONFIG } from 'constants'

import { useTextSearchService } from "features/text-search/text-search.services";

export const useSignSearchService = () => {

    const textSearchService = useTextSearchService();

    const uploadLandmarks = async (landmarks) => {
        const lambda = new AWS.Lambda();

        const keypoints = landmarks.map(l => {
            return {
                pose_landmarks: l.poseLandmarks || [],
                face_landmarks: l.faceLandmarks || [],
                left_hand_landmarks: l.leftHandLandmarks || [],
                right_hand_landmarks: l.rightHandLandmarks || [],
            }
        });

        const params = {
            FunctionName: AWS_CONFIG.LAMBDA_SAGEMAKER_INOKER,
            Payload: JSON.stringify({
                //keypoints: keypoints
                default: keypoints
            })
        };

        return new Promise((resolve, reject) => {
            lambda.invoke(params, async (err, data) => {

                if (err) {
                    console.log(err, err.stack);
                    return reject(err);
                }

                const raw = JSON.parse(data.Payload);

                if (!raw) return [];
                if (raw.errorMessage) return reject(err);

                for (var i = 0; i < raw.length; i++) {
                    var r = raw[i];
                    if (r.gloss) {
                        const searchResult = await textSearchService.search(r.gloss, 'AiResultSearch')

                        r.results = []
                        if (searchResult && searchResult.length > 0) {
                            r.results = [searchResult[0]];
                            r.word = r.results[0].word
                        }
                    }
                }
                const result = raw.filter(r => !!r.word);
                resolve(result);
            });
        });
    };

    return {
        uploadLandmarks,
    };
};
