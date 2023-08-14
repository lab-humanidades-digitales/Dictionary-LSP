import { useState, forwardRef, useImperativeHandle, useEffect } from "react";

import PropTypes from "prop-types";

// @mui material components
import {
  Switch,
  Modal,
  Divider,
  Slide,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
} from "@mui/material";

// @mui icons
import CloseIcon from "@mui/icons-material/Close";

// Material Kit 2 React components
import MKBox from "components/MKBox";
import MKButton from "components/MKButton";
import MKTypography from "components/MKTypography";
import SignVideo from "components/UI/SignVideo";

import { useTextSearchService } from "../text-search.services";

const TextSearchModal = forwardRef((props, ref) => {
  const textSearchService = useTextSearchService();

  const [title, setTitle] = useState("");
  const [query, setQuery] = useState("");
  const [show, setShow] = useState(false);
  const [message, setMessage] = useState("");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);

  const onSearch = () => {
    if (!query) {
      setMessage("");
      return;
    }

    setSearching(true);
    textSearchService
      .search(query, "AiResultSearch")
      .then((r) => {
        if (r && r.length == 0) {
          setMessage("No se encontraron resultados");
        } else {
          setMessage();
        }
        setResults(r);
      })
      .catch((ex) => {
        setMessage("Ha ocurrido un error al obtener los resultados");
      })
      .finally(() => {
        setSearching(false);
      });
  };

  useEffect(onSearch, [query]);

  const closeModal = () => {
    setShow(false);
  };

  useImperativeHandle(ref, () => ({
    showModal(visible = true, title, query, results) {
      setShow(visible);
      if (title) setTitle(title);
      if (query) setQuery(query);
      if (results) setResults(results);
    },
  }));

  return (
    <Modal
      open={show}
      disableAutoFocus={true}
      onClose={closeModal}
      sx={{ display: "grid", placeItems: "center" }}
    >
      <Slide
        direction="down"
        in={show}
        timeout={500}
        sx={{ width: { xs: "auto", md: 500 }, margin: { xs: 1, md: 0 } }}
      >
        <MKBox
          position="relative"
          width="500px"
          display="flex"
          flexDirection="column"
          borderRadius="xl"
          bgColor="white"
          shadow="xl"
        >
          <MKBox display="flex" alignItems="center" justifyContent="space-between" p={2}>
            <MKTypography variant="h5">{title}</MKTypography>
            <CloseIcon fontSize="medium" sx={{ cursor: "pointer" }} onClick={closeModal} />
          </MKBox>
          <Divider sx={{ my: 0 }} />
          <MKBox
            p={2}
            style={{
              overflowY: "auto",
              maxHeight: "calc(100vh - 150px)",
            }}
          >
            {searching ? (
              <LinearProgress
                variant="indeterminate"
                value={searching}
                sx={{ width: "100%", overflow: "hidden", mt: "4px", height: 4 }}
              ></LinearProgress>
            ) : (
              <MKBox sx={{ width: "100%", overflow: "hidden", mt: "4px", height: 4 }}></MKBox>
            )}
            {message && (
              <Typography
                variant="h5"
                style={{ textAlign: "center", paddingTop: 5, paddingBottom: 5 }}
              >
                {message}
              </Typography>
            )}
            {results &&
              results.map((result) => (
                <>
                  {result != results[0] && <Divider />}
                  <Grid container spacing={1} key={result.key}>
                    {false && (
                      <Grid item xs={12} md={12} lg={12}>
                        <Typography gutterBottom variant="h5" component="div">
                          {result.word}
                        </Typography>
                      </Grid>
                    )}
                    <Grid item xs={12} md={6} lg={6}>
                      <SignVideo
                        source={result.wordVideoUrl}
                        style={{ borderRadius: "8px" }}
                      ></SignVideo>
                    </Grid>
                    <Grid item xs={12} md={6} lg={6}>
                      <SignVideo
                        source={result.phraseVideoUrl}
                        style={{ borderRadius: "8px" }}
                      ></SignVideo>
                      <Typography variant="body2" color="text.secondary">
                        {result.phrase}
                      </Typography>
                    </Grid>
                  </Grid>
                </>
              ))}
          </MKBox>
          <Divider sx={{ my: 0 }} />
          <MKBox display="flex" justifyContent="space-between" p={1.5}>
            <div></div>
            <MKButton variant="gradient" color="info" onClick={closeModal}>
              Cerrar
            </MKButton>
          </MKBox>
        </MKBox>
      </Slide>
    </Modal>
  );
});

TextSearchModal.propTypes = {
  showModal: PropTypes.bool,
  query: PropTypes.String,
};

export default TextSearchModal;
