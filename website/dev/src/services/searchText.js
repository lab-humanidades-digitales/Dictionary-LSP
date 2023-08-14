import React, { Component } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearch } from '@fortawesome/free-solid-svg-icons';
import { Row, Col } from 'react-simple-flex-grid';
import VideoCard from '../components/VideoCard';

/* css */
import "react-simple-flex-grid/lib/main.css";
import '../css/busquedaTexto.css';


class SearchText extends Component {

  state = {
    isProcessing: false,
    allVideos: []
  };

  handleClickEvent = async ()  => {

    if (this.state.isProcessing) {
      return; // Already processing, ignore the click
    }

    this.setState({ isProcessing: true });
    this.setState({ allVideos:[] });

    const makeUrl = filename => `https://isolatedsigns.s3.amazonaws.com/${encodeURI(filename)}`;
    const getUrlFromNode = node => node.url.split("/")[3];
    const buildUrl = node => makeUrl(getUrlFromNode(node));
    const buildVideosSearchUrl = query => `https://cklvhhyl66.execute-api.us-east-1.amazonaws.com/?word=${query}`
    const getVideosFromServer = async query => {
      const response = await fetch(buildVideosSearchUrl(query));
      const data = await response.json();
      return data;
    };
    const makeUrlSentences = filename => `https://sentencesigns.s3.amazonaws.com/${encodeURI(filename)}`;
    const getUrlFromNodeSentences = node => node.urlSentence.split("/")[3];
    const buildUrlSentence = node => makeUrlSentences(getUrlFromNodeSentences(node));
    const getQueryString = () => document.getElementById("inputSearch").value;
    const mapValues = nodes => nodes.map(node => ({label: node.sign_gloss, imageUrl: buildUrl(node)}));
    const mapValuesSentences = nodes => nodes.map(node => ({label: node.text, imageUrl: buildUrlSentence(node)}));
  
    let queryString = getQueryString();
    queryString = queryString.replace(/\s+/g, " ").trim();
    queryString = queryString === " "? "": queryString;
    
    if(queryString){
      const videos = await getVideosFromServer(queryString);
      const mappedVideos = mapValues(videos);
      
      const sentences = await getVideosFromServer(queryString);
      const mappedSentences = mapValuesSentences(sentences);

      this.setState({ allVideos:[...mappedVideos,...mappedSentences] });

      this.setState({ isProcessing: false });
    }else{
      this.setState({ isProcessing: false });
    }
    
  };

  render() {
    const { isProcessing, allVideos } = this.state;  
    return (
      <div className="BusquedaTexto__page">
        
        <div className="BusquedaTexto__lsp-search">
          <div className="BusquedaTexto__search-options">  
            <input id="inputSearch" className="BusquedaTexto__inputSearch" type="text" placeholder="Buscar"/>
            <button id="inputSearchButton" className="BusquedaTexto__inputSearchButton" disabled={isProcessing }>
              <FontAwesomeIcon icon={faSearch} />
            </button>
          </div>
        </div>
        
        <div className='BusquedaTexto__container-grid'>
          <Row gutter={50}>
              {allVideos.map((video, index) =>(
                <Col  key={index}
                xs={{ span: 12 }}
                sm={{ span: 6 }}
                md={{ span: 4 }}
                lg={{ span: 3 }}
                xl={{ span: 2 }}
                >
                  <VideoCard key={index} video={video} />
                </Col>
              ))} 
          </Row>           

        </div>
      
      </div>
    );
  }

  componentDidMount() {
    const element = document.getElementById('inputSearchButton');
    element.addEventListener('click', this.handleClickEvent);
  }
  
  componentWillUnmount() {
    const element = document.getElementById('inputSearchButton');
    element.removeEventListener('click', this.handleClickEvent);
  }
}

export default SearchText;








