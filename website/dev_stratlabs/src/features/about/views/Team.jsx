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
  Link,
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

import TeamMemberCard from "../components/TeamMemberCard";
import PageHeaderContainer from "components/UI/PageHeaderContainer";
import PageContentContainer from "components/UI/PageContentContainer";

import sinImagen from "assets/images/team/imagen-defecto.jpg";

import gissellaBejarano from "assets/images/team/gissella-bejarano.jpg";
import miguelRodriguez from "assets/images/team/miguel-rodriguez.jpg";
import jennyVega from "assets/images/team/jenny-vega.jpg";
import alexandraArnaiz from "assets/images/team/alexandra-arnaiz.jpg";
import carlosVasquez from "assets/images/team/carlos-vasquez.jpg";
import cesarRamos from "assets/images/team/cesar-ramos.jpg";
import franciscoCerna from "assets/images/team/francisco-cerna.jpg";
import joeHuamani from "assets/images/team/joe-huamani.jpg";
import juanVillamonte from "assets/images/team/juan-villamonte.jpg";
import julioMendoza from "assets/images/team/julio-mendoza.jpg";
import sabinaOporto from "assets/images/team/sabina-oporto.jpg";

import stratlabs from "assets/images/team/stratlabs.png";

function Team() {
  const seccionA = [
    {
      nombre: "Gissella Bejarano",
      rol: "Líder de Inteligencia Artificial",
      link: "https://www.linkedin.com/in/gissemari/",
      foto: gissellaBejarano,
    },
    {
      nombre: "Miguel Rodriguez",
      rol: "Líder de Linguística",
      link: "",
      foto: miguelRodriguez,
    },
  ];

  const seccionB = [
    {
      nombre: "Joe Huamaní",
      rol: "Ingeniero de Aprendizaje de Máquina",
      link: "",
      foto: joeHuamani,
    },
    {
      nombre: "Sabina Oporto",
      rol: "Gestora de Anotaciones LSP y Voluntarios",
      link: "https://www.linkedin.com/in/sabinaoporto/",
      foto: sabinaOporto,
    },
    {
      nombre: "Carlos Vasquez",
      rol: "Ingeniero de Datos",
      link: "https://www.linkedin.com/in/cvasquezroque/",
      foto: carlosVasquez,
    },
    {
      nombre: "Francisco Cerna",
      rol: "Apoyo de Anotación y protocolo de pruebas",
      link: "https://linktr.ee/frantciscoch/",
      foto: franciscoCerna,
    },
    {
      nombre: "César Ramos",
      rol: "Apoyo de Anotación",
      link: "",
      foto: cesarRamos,
    },
    {
      nombre: "Juan Villamonte",
      rol: "Consultor LSP",
      link: "",
      foto: juanVillamonte,
    },
    {
      nombre: "Alexandra Arnaiz",
      rol: "Intérprete de Lengua de Señas Peruana",
      link: "",
      foto: alexandraArnaiz,
    },
  ];

  const seccionC = [
    {
      nombre: "Julio Mendoza",
      rol: "Ingeniero de la Nube",
      link: "https://www.linkedin.com/in/juliomendozah/",
      foto: julioMendoza,
    },
    {
      nombre: "Jenny Vega",
      rol: "Asesora en AWS",
      link: "",
      foto: jennyVega,
    },
    {
      nombre: "Daniel Carbajal",
      rol: "Asesor en AWS y de Desarrollo de Software",
      link: "",
      foto: sinImagen,
    },
    {
      nombre: "Corrado Daly",
      rol: "Soporte de la Nube",
      link: "",
      foto: sinImagen,
    },
    {
      nombre: "Rossy Latorraca",
      rol: "Soporte de Interfaz Web",
      link: "",
      foto: sinImagen,
    },
  ];

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
          Equipo
        </MKTypography>
      </PageHeaderContainer>

      <Container sx={{ mb: 12, mt: 3 }}>
        <Grid
          container
          spacing={3}
          sx={{ mb: 3 }}
          style={{ display: "flex", justifyContent: "center" }}
        >
          {seccionA.map((member) => (
            <Grid
              item
              xs={12}
              lg={6}
              md={6}
              xl={4}
              key={member.nombre}
              style={{ display: "flex", justifyContent: "center" }}
            >
              <MKBox mb={0} style={{ display: "flex" }}>
                <TeamMemberCard
                  image={member.foto}
                  name={member.nombre}
                  position={{ color: "info", label: member.rol }}
                  description=""
                  link={member.link}
                />
              </MKBox>
            </Grid>
          ))}
        </Grid>
        <Grid container spacing={3} style={{ display: "flex", justifyContent: "center" }}>
          {seccionB.map((member) => (
            <Grid
              item
              xs={12}
              lg={6}
              md={6}
              xl={4}
              key={member.nombre}
              style={{ display: "flex", justifyContent: "center" }}
            >
              <MKBox mb={0} style={{ display: "flex" }}>
                <TeamMemberCard
                  image={member.foto}
                  name={member.nombre}
                  position={{ color: "info", label: member.rol }}
                  description=""
                  link={member.link}
                />
              </MKBox>
            </Grid>
          ))}
        </Grid>
      </Container>

      <Container sx={{ mb: 10 }}>
        <Grid container>
          <Grid item xs={12} md={12} sx={{ mb: 6 }}>
            <MKTypography variant="h3" color="white" textAlign="center">
              Voluntarios
            </MKTypography>
          </Grid>
        </Grid>
        <Grid container spacing={3} style={{ display: "flex", justifyContent: "center" }}>
          {seccionC.map((member) => (
            <Grid
              item
              xs={12}
              lg={6}
              md={6}
              xl={4}
              key={member.nombre}
              style={{ display: "flex", justifyContent: "center" }}
            >
              <MKBox mb={0} style={{ display: "flex" }}>
                <TeamMemberCard
                  image={member.foto}
                  name={member.nombre}
                  position={{ color: "info", label: member.rol }}
                  description=""
                  link={member.link}
                />
              </MKBox>
            </Grid>
          ))}
        </Grid>

        <Grid
          container
          spacing={3}
          style={{ display: "flex", justifyContent: "center" }}
          sx={{ mt: 1 }}
        >
          <Grid
            item
            xs={12}
            md={4}
            lg={3}
            xl={2}
            style={{ display: "flex", justifyContent: "center" }}
          >
            <MKBox mb={0} style={{ display: "flex" }}>
              <Card sx={{ mt: 0 }} style={{ backgroundColor: "#ffffff15" }}>
                <Grid container>
                  <Grid item xs={12} md={12} lg={12} sx={{}}>
                    <MKBox width="100%" pt={2} pb={1} px={2}>
                      <Link href="https://stratlabs.pe" target="_blank">
                        <MKBox component="img" src={stratlabs} alt={"StratLabs"} width="100%" />
                      </Link>
                    </MKBox>
                  </Grid>
                </Grid>
              </Card>
            </MKBox>
          </Grid>
        </Grid>
      </Container>
    </>
  );
}

export default Team;
