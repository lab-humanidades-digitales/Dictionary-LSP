// @mui icons
import FacebookIcon from "@mui/icons-material/Facebook";
import TwitterIcon from "@mui/icons-material/Twitter";
import GitHubIcon from "@mui/icons-material/GitHub";
import YouTubeIcon from "@mui/icons-material/YouTube";

// Material Kit 2 React components
import MKTypography from "components/MKTypography";

export default {
  company: { href: "https://www.pucp.edu.pe/", name: "PUCP" },
  links: [
    {
      name: "Búsqueda en Español",
      route: "/search-by-text",
    },
    {
      name: "Búsqueda por Seña",
      route: "/search-by-sign",
    },
    {
      name: "Equipo",
      route: "/team",
    },
  ],
  socials: [
    { icon: <GitHubIcon fontSize="small" />, link: "https://github.com/gissemari/PeruvianSignLanguage" },
  ],
  light: false,
};
