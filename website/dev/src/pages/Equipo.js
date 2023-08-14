import React, { lazy, Suspense } from 'react';

import '../css/equipo.css'

const Header = lazy(() => import('../components/Header'));
const Footer = lazy(() => import('../components/Footer'));

function Equipo() {

  return (
    <div className="Equipo">
      <Suspense fallback={<div>Loading...</div>}>
        <Header />
      </Suspense>
      equipo
      <Suspense fallback={<div>Loading...</div>}>
        <Footer />
      </Suspense>
    </div>
  );
}

export default Equipo;
