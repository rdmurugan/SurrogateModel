import { createTheme } from '@mui/material/styles';

// ANSYS/ProE Professional Color Palette
const professionalColors = {
  // ANSYS-style dark theme
  primaryDark: '#1e1e1e',        // Main background
  secondaryDark: '#2d2d30',      // Panel backgrounds
  tertiaryDark: '#3e3e42',       // Elevated surfaces
  quaternaryDark: '#4d4d50',     // Borders and dividers

  // ANSYS blue accents
  ansysBlue: '#00a8ff',          // Primary blue
  ansysLightBlue: '#4fc3f7',     // Light blue accents
  ansysDarkBlue: '#0077cc',      // Dark blue

  // ProE orange accents
  proeOrange: '#ff7043',         // ProE signature orange
  proeLightOrange: '#ffab91',    // Light orange
  proeDarkOrange: '#d84315',     // Dark orange

  // Professional grays
  coolGray: '#6c757d',           // Text secondary
  warmGray: '#868e96',           // Disabled text
  lightGray: '#e9ecef',          // Light borders

  // Status colors
  successGreen: '#28a745',       // Success states
  warningYellow: '#ffc107',      // Warning states
  errorRed: '#dc3545',           // Error states
  infoBlue: '#17a2b8',          // Info states

  // Text colors
  textPrimary: '#ffffff',        // Primary text
  textSecondary: '#b0b0b0',      // Secondary text
  textDisabled: '#757575',       // Disabled text
  textContrast: '#000000',       // Contrast text
};

export const professionalTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: professionalColors.ansysBlue,
      light: professionalColors.ansysLightBlue,
      dark: professionalColors.ansysDarkBlue,
      contrastText: professionalColors.textPrimary,
    },
    secondary: {
      main: professionalColors.proeOrange,
      light: professionalColors.proeLightOrange,
      dark: professionalColors.proeDarkOrange,
      contrastText: professionalColors.textPrimary,
    },
    background: {
      default: professionalColors.primaryDark,
      paper: professionalColors.secondaryDark,
    },
    // Custom surface colors (extending Material-UI types)
    //@ts-ignore
    surface: {
      main: professionalColors.tertiaryDark,
      light: professionalColors.quaternaryDark,
    },
    text: {
      primary: professionalColors.textPrimary,
      secondary: professionalColors.textSecondary,
      disabled: professionalColors.textDisabled,
    },
    success: {
      main: professionalColors.successGreen,
    },
    warning: {
      main: professionalColors.warningYellow,
    },
    error: {
      main: professionalColors.errorRed,
    },
    info: {
      main: professionalColors.infoBlue,
    },
    divider: professionalColors.quaternaryDark,
  },
  typography: {
    fontFamily: '"Segoe UI", "Roboto", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1.125rem',
      fontWeight: 500,
    },
    body1: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
    body2: {
      fontSize: '0.75rem',
      lineHeight: 1.4,
    },
    button: {
      fontSize: '0.875rem',
      fontWeight: 500,
      textTransform: 'none',
    },
    caption: {
      fontSize: '0.75rem',
      fontWeight: 400,
    },
  },
  shape: {
    borderRadius: 2, // Minimal rounded corners like ANSYS
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarWidth: 'thin',
          scrollbarColor: `${professionalColors.quaternaryDark} ${professionalColors.secondaryDark}`,
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: professionalColors.secondaryDark,
          },
          '&::-webkit-scrollbar-thumb': {
            background: professionalColors.quaternaryDark,
            borderRadius: '4px',
            '&:hover': {
              background: professionalColors.coolGray,
            },
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: professionalColors.primaryDark,
          borderBottom: `1px solid ${professionalColors.quaternaryDark}`,
          boxShadow: 'none',
        },
      },
    },
    MuiToolbar: {
      styleOverrides: {
        root: {
          minHeight: '48px !important',
          padding: '0 16px',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '2px',
          textTransform: 'none',
          minHeight: '28px',
          padding: '4px 12px',
          border: `1px solid ${professionalColors.quaternaryDark}`,
          '&:hover': {
            borderColor: professionalColors.ansysBlue,
          },
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: `0 0 0 1px ${professionalColors.ansysBlue}`,
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: professionalColors.secondaryDark,
          border: `1px solid ${professionalColors.quaternaryDark}`,
        },
        elevation1: {
          boxShadow: 'none',
        },
        elevation2: {
          boxShadow: `0 1px 3px ${professionalColors.primaryDark}`,
        },
        elevation3: {
          boxShadow: `0 2px 6px ${professionalColors.primaryDark}`,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: professionalColors.secondaryDark,
          border: `1px solid ${professionalColors.quaternaryDark}`,
          borderRadius: '2px',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: professionalColors.quaternaryDark,
            },
            '&:hover fieldset': {
              borderColor: professionalColors.ansysBlue,
            },
            '&.Mui-focused fieldset': {
              borderColor: professionalColors.ansysBlue,
            },
          },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          minHeight: '32px',
          fontSize: '0.875rem',
          borderBottom: `1px solid ${professionalColors.quaternaryDark}`,
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          minHeight: '32px',
          '& .MuiTabs-indicator': {
            height: '2px',
            backgroundColor: professionalColors.ansysBlue,
          },
        },
      },
    },
  },
});

export default professionalTheme;