import axios from 'axios'
import { API_URLS } from 'constants'


const API = axios.create({
  baseURL: API_URLS.WORDS,
  // timeout: 1000,
  // headers: {'X-Custom-Header': 'foobar'}
  validateStatus: function (status) {
    return status < 500; // Resolve only if the status code is less than 500
  },
})

API.GetWordUrl = (filename) => {
  if (filename.startsWith('s3://'))
    return filename.replace('s3://isolatedsigns/', API_URLS.ISOLATED_SIGNS);

  return `${API_URLS.ISOLATED_SIGNS}/${encodeURI(filename)}`
}

API.GetPhraseUrl = (filename) => {
  if (filename.startsWith('s3://'))
    return filename.replace('s3://sentencesigns/', API_URLS.SENTENCE_SIGNS);

  return `${API_URLS.SENTENCE_SIGNS}/${encodeURI(filename)}`
}


API.interceptors.request.use(
  config => {

    const accessToken = localStorage.getItem('jwt');
    if (accessToken && config && config.headers) {
      config.headers.Authorization = `Bearer ${accessToken}`
      // console.log('Axios call with JWT: ' + config.url)
      // console.log(accessToken)
    }
    else {
      // console.log('Axios call without JWT: ' + config.url)
    }
    // console.log(config);
    return config
  },
  error => {
    // console.log('Axios error');
    // console.log(error);
    return Promise.reject(error)
  },
)

API.interceptors.response.use(
  response => {
    // console.log('Axios response');
    // console.log(response);
    return response;
  },
  error => {
    // console.log('Axios response error');
    // console.log(error);
    return Promise.reject(error);
  },
)

export default API
