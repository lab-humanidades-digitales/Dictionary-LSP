import React from "react";
import PropTypes from "prop-types";
// @mui material components
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";

// Material Kit 2 React components
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";

// Material Kit 2 React examples
import DefaultNavbar from "./DefaultNavbar";
import CenteredFooter from "./CenteredFooter";

// Routes
import navbarRoutes from "navbar.routes";
import footerRoutes from "footer.routes";

// Images
//import bgImage from "assets/images/bg-about-us.jpg";
import bgImage from "assets/images/shapes/waves-white-2.svg";

DefaultLayout.propTypes = {
  children: PropTypes.node.isRequired,
};

function DefaultLayout({ children }) {
  return (
    <>
      <DefaultNavbar
        routes={navbarRoutes}
        /* action={{
          type: "external",
          route: "https://www.creative-tim.com/product/material-kit-react",
          label: "free download",
          color: "default",
        }}*/
        transparent
        light
      />

      <MKBox
        sx={{
          minHeight: "100vh",
          backgroundImage: ({ functions: { linearGradient, rgba }, palette: { gradients } }) =>
            `${linearGradient(rgba("#042552", 1), rgba("#153694", 0.9))}, url(${bgImage})`,
          display: "grid",
          placeItems: "center",
        }}
      >
        {children}

        <MKBox pt={0} px={1} mt={0} light>
          <CenteredFooter {...footerRoutes} light />
        </MKBox>
      </MKBox>
    </>
  );
}

export default DefaultLayout;
