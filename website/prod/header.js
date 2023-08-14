import React from 'react';

const Header = ({ children }) => (
  <header style="color: #163297;" >
    
    <nav class="navbar navbar-default navbar-static-top">
      <h1 class="d-flex align-items-center fs-4 text-white mb-0">
        <img src="images/logo_pucp_normal.png" width="20%" height="30%" class="me-3" alt="PUCP">
        DICCIONARIO DE LSP AL ESPAÑOL
      </h1>
    </nav>

    <div style="background-color: #042354" class="d-flex justify-content-center py-3">
      <div class="col-md-5 text-center">
        {children}
      </div>
    </div>
  </header>
);

export default Header;