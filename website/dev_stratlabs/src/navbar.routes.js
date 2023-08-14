// @mui material components
import Icon from "@mui/material/Icon";

// @mui icons
import SearchIcon from '@mui/icons-material/Search';
import SignLanguageIcon from '@mui/icons-material/SignLanguage';
import GitHubIcon from '@mui/icons-material/GitHub';
// eslint-disable-next-line no-unused-vars
const routes = [
  {
    name: "Búsqueda en Español",
    icon: <SearchIcon />,
    route: "/search-by-text",
  },
  {
    name: "Búsqueda por Seña",
    icon: <SignLanguageIcon />,
    route: "/search-by-sign",
  }
  ,
  {
    name: "Equipo",
    route: "/team",
  }
];

export default routes;
