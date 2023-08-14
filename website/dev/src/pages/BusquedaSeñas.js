import React, { lazy, Suspense } from 'react';
import '../css/busquedaSeñas.css';

const Header = lazy(() => import('../components/Header'));
const Footer = lazy(() => import('../components/Footer'));

const SearchSign = lazy(() => import('../services/searchSign'));


function BusquedaSeñas() {
  return (
    <div className='BusquedaSeña__all'>
      <Suspense fallback={<div>Loading...</div>}>
        <Header />
      </Suspense>
      <div className="BusquedaSeña__container">
        <Suspense fallback={<div>Loading...</div>}>
          <SearchSign />
        </Suspense>
        {/* <button >id="SentencesSearchButton" className="lsp-btn">Mostrar oraciones</button> */}
      </div>
      <Suspense fallback={<div>Loading...</div>}>
        <Footer />
      </Suspense>
    </div>
  );
}

export default BusquedaSeñas;

