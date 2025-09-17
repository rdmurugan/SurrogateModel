import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Container, ThemeProvider, CssBaseline, Box } from '@mui/material';
import ProfessionalNavigation from './components/ProfessionalNavigation';
import ProfessionalStatusBar from './components/ProfessionalStatusBar';
import ModelTreePanel from './components/ModelTreePanel';
import Dashboard from './pages/Dashboard';
import Datasets from './pages/Datasets';
import Models from './pages/Models';
import Predictions from './pages/Predictions';
import NextGenML from './pages/NextGenML';
import Login from './pages/Login';
import AdvancedVisualizations from './components/AdvancedVisualizations';
import { AuthProvider, useAuth } from './services/AuthContext';
import { UIStateProvider } from './contexts/UIStateContext';
import { ProjectProvider } from './contexts/ProjectContext';
import professionalTheme from './theme/professionalTheme';

function App() {
  return (
    <ThemeProvider theme={professionalTheme}>
      <CssBaseline />
      <AuthProvider>
        <UIStateProvider>
          <ProjectProvider>
            <Box sx={{
              display: 'flex',
              flexDirection: 'column',
              height: '100vh',
              backgroundColor: '#1e1e1e',
              overflow: 'hidden'
            }}>
              <ProfessionalNavigation />

              <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'hidden' }}>
                <ProfessionalWorkspace />
              </Box>

              <ProfessionalStatusBar />
            </Box>
          </ProjectProvider>
        </UIStateProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

function ProfessionalWorkspace() {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return (
      <Box sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
        height: '100%',
        backgroundColor: '#1e1e1e'
      }}>
        <Login />
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', width: '100%', height: '100%' }}>
      {/* Left Sidebar - Model Tree */}
      <Box sx={{
        width: '280px',
        minWidth: '280px',
        backgroundColor: '#2d2d30',
        borderRight: '1px solid #3e3e42',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <ModelTreePanel />
      </Box>

      {/* Main Content Area */}
      <Box sx={{
        flexGrow: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        <Box sx={{
          flexGrow: 1,
          overflow: 'auto',
          backgroundColor: '#1e1e1e',
          padding: '0'
        }}>
          <Routes>
            <Route path="/login" element={<Navigate to="/" replace />} />
            <Route path="/" element={<Dashboard />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/models" element={<Models />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/nextgen-ml" element={<NextGenML />} />
            <Route path="/visualizations" element={<AdvancedVisualizations />} />
          </Routes>
        </Box>
      </Box>
    </Box>
  );
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

export default App;