const makeUrl = filename => `https://isolatedsigns.s3.amazonaws.com/${encodeURI(filename)}`;
const getUrlFromNode = node => node.url.split("/")[3];
const buildUrl = node => makeUrl(getUrlFromNode(node));
const buildVideosSearchUrl = query => `https://cklvhhyl66.execute-api.us-east-1.amazonaws.com/?word=${query}`
const getVideosFromServer = query => fetch(buildVideosSearchUrl(query)).then(response => response.json())
const makeUrlSentences = filename => `https://sentencesigns.s3.amazonaws.com/${encodeURI(filename)}`;
const getUrlFromNodeSentences = node => node.urlSentence.split("/")[3];
const buildUrlSentence = node => makeUrlSentences(getUrlFromNodeSentences(node));
const getQueryString = () => document.getElementById("inputSearch").value;
const mapValues = nodes => nodes.map(node => ({label: node.sign_gloss, imageUrl: buildUrl(node)}));
const mapValuesSentences = nodes => nodes.map(node => ({label: node.text, imageUrl: buildUrlSentence(node)}));
const getListElement = () => document.getElementById("products-list");
const getListSentence = () => document.getElementById("sentences-list");
const buildVideoNode = ({ label, imageUrl}) =>
`
    <div class="col-sm-6 col-md-4 col-lg-3 mb-4 ml-3">
        <div class="card h-100 shadow rounded" style="margin-right: 20px;" category="adjectives">
            <video class="card-img-top" height="205px" width="205px" controls>
                <source src="${imageUrl}" type="video/mp4">
            </video>
            <div class="card-body rounded-bottom" style="background-color: #163297">
                <h5 class="card-title text-white">${label}</h5>
            </div>
        </div>
    </div>
`;

const appendVideo = node => {
    getListElement().insertAdjacentHTML('beforeend', buildVideoNode(node))
}

document.getElementById("inputSearchButton").addEventListener("click", () => {
    const queryString = getQueryString();
    getListElement().innerHTML = "";
    getVideosFromServer(queryString).then(mapValues).then(nodes => nodes.map(appendVideo));
    getListSentence().innerHTML = "";
    getVideosFromServer(queryString).then(mapValuesSentences).then(nodes => nodes.map(appendVideo))
})

