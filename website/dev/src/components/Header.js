import React from 'react';
import logo from '../assets/logo_pucp_small.png';
import '../css/header.css'; // Import the CSS file
import { Outlet, Link } from "react-router-dom";

const Header = () => {

    return (
        <header className="header">
            <nav className="navbar ">
                <h1 className="header__title" >
                    <img src={logo} className="header__logo" alt="PUCP" />
                        DICCIONARIO DE LSP AL ESPAÑOL
                </h1>
            </nav>

            <div className="header__container">
                <div className="header__buttons">
                    <Link className="button" to="/busquedaTexto" role="button">
                            Búsqueda en Español
                    </Link>
                    <Link className="button" to="/busquedaSeñas" role="button">           
                            Búsqueda por Señas
                    </Link>
                    <Link className="button" to="/equipo" role="button">
                                Equipo
                    </Link>
                    
                </div>
            </div>
            <Outlet />
        </header>
    );
};

export default Header;
