import React from 'react';
import { createRoot } from 'react-dom/client';
import reportWebVitals from './reportWebVitals';
import { BrowserRouter, Routes, Route} from "react-router-dom";

import BusquedaTexto from './pages/BusquedaTexto';
import BusquedaSeñas from "./pages/BusquedaSeñas";
import Equipo from "./pages/Equipo";
import ErrorPage from "./pages/Error";



import './css/index.css';


const root = createRoot(document.getElementById('root'));

root.render(
  
    <BrowserRouter>
      <Routes>

        <Route index element={<BusquedaTexto />} />
        <Route path="busquedaTexto" element={<BusquedaTexto />} />
        <Route path="busquedaSeñas" element={<BusquedaSeñas />} />
        <Route path="equipo" element={<Equipo />} />
        <Route path="error" element={<ErrorPage />} />
        <Route path="*" element={<ErrorPage />} />
        
      </Routes>
    </BrowserRouter>

);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
