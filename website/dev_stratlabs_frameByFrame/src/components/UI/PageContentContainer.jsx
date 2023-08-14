import React from "react";
import PropTypes from "prop-types";

import { Card, Container } from "@mui/material";

DefaultLayout.propTypes = {
  children: PropTypes.node.isRequired,
  sx: PropTypes.object,
};

function DefaultLayout({ children, sx }) {
  return (
    <>
      <Container>
        <Card
          sx={{
            p: 2,
            mx: { xs: 2, lg: 3 },
            mt: 4,
            mb: 4,
            boxShadow: ({ boxShadows: { xxl } }) => xxl,
            ...sx,
          }}
        >
          {children}
        </Card>
      </Container>
    </>
  );
}

export default DefaultLayout;
