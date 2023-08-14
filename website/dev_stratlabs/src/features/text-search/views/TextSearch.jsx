import React, { useState, useCallback, useEffect } from "react";

// @mui material components
import {
  Container,
  Grid,
  Card,
  InputAdornment,
  CardMedia,
  CardActionArea,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Divider,
  Autocomplete,
  TextField,
} from "@mui/material";

import parse from "autosuggest-highlight/parse";
import match from "autosuggest-highlight/match";

// Material Kit 2 React components
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";
import MKInput from "components/MKInput";

// @mui icons
import SearchIcon from "@mui/icons-material/Search";

import SignVideo from "components/UI/SignVideo";
import PageHeaderContainer from "components/UI/PageHeaderContainer";
import PageContentContainer from "components/UI/PageContentContainer";

import { styled } from "@mui/material/styles";
import { useNotification } from "contextProviders/NotificationContext";
import { useTextSearchService } from "../text-search.services";
import { DICTIONARY_WORD_LIST } from "utils/dictionary-words";

const SearchTextField = styled(MKInput)({
  "& label.Mui-focused": {
    color: "#A0AAB4",
  },
  "& .MuiInput-underline:after": {
    borderBottomColor: "#B2BAC2",
  },
  "& .MuiOutlinedInput-root": {
    backgroundColor: "white",
    "& fieldset": {
      borderColor: "#E0E3E7",
    },
    "&:hover fieldset": {
      borderColor: "#B2BAC2",
    },
    "&.Mui-focused fieldset": {
      borderColor: "#6F7E8C",
    },
  },
});

function TextSearch() {
  const textSearchService = useTextSearchService();
  const notification = useNotification();

  const [query, setQuery] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [message, setMessage] = useState("");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);

  const onSearch = () => {
    if (!searchQuery) {
      setMessage("Ingrese una palabra para realizar la búsqueda");
      return;
    }

    setSearching(true);
    textSearchService
      .search(searchQuery, "wordSearch")
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

        /* notification.open({
          title: "Ha ocurrido un error",
          subTitle: "",
          type: "error",
        });*/
      })
      .finally(() => {
        setSearching(false);
      });
  };

  useEffect(() => {
    if (query.length < 1) SetAutocompleteOpen(false);
  }, [query]);
  useEffect(onSearch, [searchQuery]);

  const [isAutocompleteOpen, SetAutocompleteOpen] = React.useState(false);

  const handleAutocompleteOpen = () => {
    SetAutocompleteOpen(query.length > 0);
  };

  const filterOptions = (options, { inputValue }) => {
    var val = inputValue.toLowerCase();
    var filtered = options.filter((x) => x.label.toLowerCase().indexOf(val) > -1);

    if (filtered.length == 0) SetAutocompleteOpen(false);

    return filtered;
  };

  return (
    <>
      <PageHeaderContainer>
        <MKTypography
          variant="h1"
          color="white"
          mt={-6}
          mb={1}
          sx={({ breakpoints, typography: { size } }) => ({
            [breakpoints.down("md")]: {
              fontSize: size["3xl"],
            },
          })}
        >
          Búsqueda en Español
        </MKTypography>
        <MKTypography
          variant="body1"
          color="white"
          textAlign="center"
          px={{ xs: 6, lg: 12 }}
          mt={1}
          mb={5}
        >
          Escribe la palabra que desees encontrar en Lengua de Señas Peruana.
        </MKTypography>
        <div>
          <Autocomplete
            open={isAutocompleteOpen}
            onOpen={handleAutocompleteOpen}
            onClose={() => SetAutocompleteOpen(false)}
            freeSolo={true}
            options={DICTIONARY_WORD_LIST}
            forcePopupIcon={false}
            //getOptionLabel={(option) => option.label}
            filterOptions={filterOptions}
            onChange={(e, newValue) => {
              if (newValue) {
                if (newValue.label) {
                  setQuery(newValue.label);
                  setSearchQuery(newValue.label);
                } else {
                  setQuery(newValue);
                  setSearchQuery(newValue);
                }
              }
            }}
            renderInput={(params) => (
              <SearchTextField
                {...params}
                sx={{
                  width: {
                    xs: "calc(100vw - 80px)",
                    sm: 400,
                    md: 470,
                  },
                }}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                }}
                /*onKeyDown={(e) => {
                  // eslint-disable-next-line no-debugger
                  if (e.key === "Enter") setSearchQuery(query);
                }}*/
                variant="outlined"
                type="text"
                InputProps={{
                  ...params.InputProps,
                  style: { paddingRight: 13, paddingLeft: 17 },
                  sx: { fontSize: "x-large" },
                  startAdornment: (
                    <InputAdornment>
                      <SearchIcon />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment>
                      <MKButton
                        variant="contained"
                        color="light"
                        onClick={() => setSearchQuery(query)}
                        loading={true}
                      >
                        Buscar
                      </MKButton>
                    </InputAdornment>
                  ),
                }}
              />
            )}
            renderOption={(props, option, { inputValue }) => {
              if (!inputValue || inputValue.length == 0) return null;

              const matches = match(option.label, inputValue, { insideWords: true });
              const parts = parse(option.label, matches);

              if (parts && parts.length == 0) return null;
              return (
                <>
                  {parts && parts.length > 0 && (
                    <li {...props}>
                      <div>
                        {parts.map((part, index) => (
                          <span
                            key={index}
                            style={{
                              textDecoration: part.highlight ? "underline" : "normal",
                            }}
                          >
                            {part.text}
                          </span>
                        ))}
                      </div>
                    </li>
                  )}
                </>
              );
            }}
          />

          {searching ? (
            <LinearProgress
              variant="indeterminate"
              value={searching}
              sx={{ width: "100%", overflow: "hidden", mt: "4px", height: 4 }}
            ></LinearProgress>
          ) : (
            <MKBox sx={{ width: "100%", overflow: "hidden", mt: "4px", height: 4 }}></MKBox>
          )}
        </div>
      </PageHeaderContainer>

      <Container>
        {message && (
          <Typography
            variant="h5"
            color="#fff"
            style={{ textAlign: "center", marginBottom: "100px" }}
          >
            {message}
          </Typography>
        )}
        <Grid container spacing={3} style={{ flexDirection: "column", alignItems: "center" }}>
          {results &&
            results.map((result) => (
              <>
                <Grid item xs={12} md={12} lg={6} key={result.key}>
                  <Card sx={{ padding: 2 }}>
                    <Grid container>
                      <Grid item xs={12} md={12} lg={12}>
                        <Typography gutterBottom variant="h5" component="div">
                          {result.word}
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={6} lg={6}>
                        <SignVideo
                          source={result.wordVideoUrl}
                          style={{ borderRadius: "8px" }}
                        ></SignVideo>
                      </Grid>
                      <Grid item xs={6} md={6} lg={6}>
                        <SignVideo
                          source={result.phraseVideoUrl}
                          style={{ borderRadius: "8px" }}
                        ></SignVideo>
                        <Typography variant="body2" color="text.secondary">
                          {result.phrase}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Card>
                </Grid>
              </>
            ))}
        </Grid>

        {false && (
          <Grid container spacing={3} alignItems="center">
            {results &&
              results.map((result) => (
                <>
                  <Grid item xs={12} md={6} lg={4} key={result.key}>
                    {true && (
                      <Card>
                        <CardContent sx={{ mt: 2 }}>
                          <Typography gutterBottom variant="h5" component="div">
                            {result.word}
                          </Typography>
                          <SignVideo
                            source={result.wordVideoUrl}
                            style={{ borderRadius: "8px" }}
                          ></SignVideo>
                          <Divider />
                          <Typography variant="body2" color="text.secondary">
                            {result.phrase}
                          </Typography>
                          <SignVideo
                            source={result.phraseVideoUrl}
                            style={{ borderRadius: "8px" }}
                          ></SignVideo>
                        </CardContent>
                      </Card>
                    )}
                    {false && (
                      <Card sx={{}}>
                        <CardActionArea>
                          <CardContent>
                            <Typography gutterBottom variant="h5" component="div">
                              {result.word}
                            </Typography>

                            <video
                              style={{ borderRadius: "0.75rem 0.75rem 0 0", width: "100%" }}
                              src={result.wordVideoUrl}
                              loop={true}
                              controls={true}
                            ></video>

                            <Typography gutterBottom variant="h6" component="div">
                              Uso en contexto
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {result.phrase}
                            </Typography>

                            <video
                              style={{ borderRadius: "0.75rem 0.75rem 0 0", width: "100%" }}
                              src={result.phraseVideoUrl}
                              loop={true}
                              controls={true}
                            ></video>
                          </CardContent>
                        </CardActionArea>
                      </Card>
                    )}
                  </Grid>
                </>
              ))}
          </Grid>
        )}
      </Container>
    </>
  );
}

export default TextSearch;
