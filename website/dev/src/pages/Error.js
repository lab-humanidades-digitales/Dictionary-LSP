import React from 'react';
import Header from '../components/Header';

import '../css/equipo.css'


function Error() {
  return (
    <div className="App">
      <Header/>
      <div>
        <h1>Link inválido</h1>
        <p>Seleccione alguna de las opciones de arriba</p>
      </div>
    </div>
  );
}

export default Error;
