import React from 'react';

/* css */
import '../css/instrucciones.css'

const Instrucciones = () => {
    return (
        <main  id="instructions" className="instructions-container"> 
        <div className="instructions-components">
            <h1 className='lead-title'>Instrucciones</h1>
            <p className="lead">Al dar clic en el botón INICIAR la cámara de tu computadora se prenderá y se mostrarán 3 recuadros para que ubiques tu rostro y las palmas de tus manos. Cuando te hayas ubicado correctamente, se mostrará un contador (3,2,1) y podrás realizar la seña a buscar en dos segundos. Recomendamos ubicarse aproximadamente a 80cm de la cámara. </p>
            <button className="init-button" id="startButton">
                Iniciar »
            </button>
        </div>
    </main>
    );
  };
export default Instrucciones;