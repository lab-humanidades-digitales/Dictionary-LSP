import { useEffect } from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";

// @mui material components
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";

// Material Kit 2 React themes
import theme from "assets/theme";

import textSearchRoutes from "features/text-search/text-search.routes";
import signSearchRoutes from "features/sign-search/sign-search.routes";
import aboutRoutes from "features/about/about.routes";

import { NotificationProvider } from 'contextProviders/NotificationContext'


export default function App() {
  const { pathname } = useLocation();

  // Setting page scroll to 0 when changing the route
  useEffect(() => {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
  }, [pathname]);

  const routes = []
    .concat(textSearchRoutes)
    .concat(signSearchRoutes)
    .concat(aboutRoutes)


  const getRoutes = (allRoutes) =>
    allRoutes.map((route) => {
      if (route.collapse) {
        return getRoutes(route.collapse);
      }

      if (route.route) {
        return <Route exact path={route.route} element={route.component} key={route.key} />;
      }
      return null;
    });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <NotificationProvider>
        <Routes>
          {getRoutes(routes)}

          <Route path="*" element={<Navigate to="/search-by-text" />} />
        </Routes>
      </NotificationProvider>
    </ThemeProvider>
  );
}
