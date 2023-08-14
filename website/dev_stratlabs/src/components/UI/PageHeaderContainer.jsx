import React from "react";
import PropTypes from "prop-types";

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import MKBox from "components/MKBox";

DefaultLayout.propTypes = {
  children: PropTypes.node.isRequired,
};

function DefaultLayout({ children }) {
  return (
    <>
      <MKBox
        width="100%"
        sx={{
          display: "grid",
          placeItems: "center",
          paddingTop: {
            lg: 30,
            xs: 25,
          },
          marginBottom: {
            lg: 5,
            xs: 3,
          },
        }}
      >
        <Container>
          <Grid
            container
            item
            xs={12}
            lg={8}
            justifyContent="center"
            alignItems="center"
            flexDirection="column"
            sx={{ mx: "auto", textAlign: "center" }}
          >
            {children}
          </Grid>
        </Container>
      </MKBox>
    </>
  );
}

export default DefaultLayout;
