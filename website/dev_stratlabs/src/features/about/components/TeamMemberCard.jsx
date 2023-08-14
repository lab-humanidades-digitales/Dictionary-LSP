import PropTypes from "prop-types";

// @mui material components
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";

// Material Kit 2 React components
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";

import GitHubIcon from "@mui/icons-material/GitHub";
import LinkedInIcon from "@mui/icons-material/LinkedIn";
import LinkIcon from "@mui/icons-material/Link";

import Link from "@mui/material/Link";

function TeamMemberCard({ image, name, position, description, link }) {
  const getLinkTag = () => {
    if (!link) return;

    if (link.indexOf("linkedin") > -1) {
      return (
        <>
          <LinkedInIcon fontSize="small" color="white" />
          <span style={{ marginLeft: 4 }}>
            {link.substring(link.indexOf("/in/") + 4).replace("/", "")}
          </span>
        </>
      );
    }
    if (link.indexOf("linktr.ee") > -1) {
      return (
        <>
          <LinkIcon fontSize="small" color="white" />
          <span style={{ marginLeft: 4 }}>
            {link.substring(link.indexOf("linktr.ee") + 10).replace("/", "")}
          </span>
        </>
      );
    }
  };

  return (
    <Card sx={{ mt: 0 }} style={{ backgroundColor: "#ffffff15" }}>
      <Grid container>
        <Grid item xs={12} md={6} lg={4} sx={{}}>
          <MKBox width="100%" pt={2} pb={1} px={2}>
            <MKBox
              component="img"
              src={image}
              alt={name}
              width="100%"
              borderRadius="md"
              shadow="lg"
            />
          </MKBox>
        </Grid>
        <Grid item xs={12} md={6} lg={8} sx={{ my: "auto" }}>
          <MKBox pt={{ xs: 1, lg: 2.5 }} pb={2.5} pr={4} pl={{ xs: 4, lg: 1 }} lineHeight={1}>
            <MKTypography variant="h5" color={"white"}>
              {name}
            </MKTypography>
            <MKTypography variant="h6" color={"light"} mb={1}>
              {position.label}
            </MKTypography>
            <MKTypography variant="body2" color="text">
              {description}

              <MKTypography
                component={Link}
                href={link}
                variant="body2"
                color={"white"}
                fontWeight="regular"
                style={{ alignItems: "center", display: "flex" }}
                rel="noopener"
                target="_blank"
                sx={{ opacity: 0.7, "&:hover": { opacity: 1 } }}
              >
                {getLinkTag()}
              </MKTypography>
            </MKTypography>
          </MKBox>
        </Grid>
      </Grid>
    </Card>
  );
}

TeamMemberCard.propTypes = {
  image: PropTypes.string.isRequired,
  name: PropTypes.string.isRequired,
  position: PropTypes.shape({
    color: PropTypes.oneOf([
      "primary",
      "secondary",
      "info",
      "success",
      "warning",
      "error",
      "dark",
      "light",
    ]),
    label: PropTypes.string.isRequired,
  }).isRequired,
  link: PropTypes.string,
  description: PropTypes.string.isRequired,
};

export default TeamMemberCard;
