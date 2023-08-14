import React, { lazy, Suspense } from 'react';

/* css */
import '../css/busquedaTexto.css';

/* Components */
const Header = lazy(() => import('../components/Header'));
const Footer = lazy(() => import('../components/Footer'));

/* services */
const SearchText = lazy(() => import('../services/searchText'));

function BusquedaTexto() {
  return (
    <div className='BusquedaTexto__all'>
      <Suspense fallback={<div>Loading...</div>}>
        <Header />
      </Suspense>
      <div className="BusquedaTexto__container">
        {/* eslint-disable-next-line */} 
        <Suspense fallback={<div>Loading...</div>}>
          <SearchText />
        </Suspense>
       
      </div>
      <Suspense fallback={<div>Loading...</div>}>
        <Footer />
      </Suspense>
    </div>
  );
}

export default BusquedaTexto;
