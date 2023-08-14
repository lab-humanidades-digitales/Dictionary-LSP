import React, {useEffect, useState} from 'react';
import Instrucciones from '../components/Instrucciones';
import WebcamRecord from '../components/WebcamRecord';

/* css */
import "react-simple-flex-grid/lib/main.css";
import '../css/busquedaTexto.css';

const SearchText = () => {
  const [isVisible, setIsVisible] = useState(false);

  // const [nombre, setNombre] = useState(null)

  const handleClickEvent = async () => {
    const instructions = document.getElementById('instructions');
    instructions.style.display = 'none';
    setIsVisible(true);
  };

  useEffect(() => {
    const startButton = document.getElementById('startButton');
    startButton.addEventListener('click', handleClickEvent);

    return () => {
      startButton.removeEventListener('click', handleClickEvent);
    };
  }, []);

  return (
    <div className="BusquedaSeña__page">
      {/* INSTRUCCIONES */}
      <Instrucciones />

      {/* WEBCAM */}
      {isVisible && <div className={'BusquedaSeña__webcam-container_on'}>
                      <div className={'BusquedaSeña__webcam-component_on'}>
                        {isVisible && <WebcamRecord/>}
                      </div>
                    </div>}

      {/* RESULTS */}
    </div>
  );
};

export default SearchText;


