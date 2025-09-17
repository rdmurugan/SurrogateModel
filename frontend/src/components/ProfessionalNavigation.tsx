import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  IconButton,
  Menu,
  MenuItem,
  Divider,
  Chip,
  ButtonGroup,
  Button,
  Paper,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  AccountCircle,
  Settings,
  Help,
  ExitToApp,
  Save,
  FolderOpen,
  Refresh,
  ViewInAr,
  Timeline,
  Analytics,
  Build,
  Science,
  Engineering,
  Memory,
  Speed,
  BugReport,
  CloudDownload,
  Print,
  Share,
  GridOn,
  Layers,
  Visibility,
  VisibilityOff,
  ZoomIn,
  ZoomOut,
  CenterFocusStrong,
  RotateLeft,
  RotateRight,
  PlayArrow,
  Pause,
  Stop,
  SkipNext,
  SkipPrevious,
  Add,
  TrendingUp,
} from '@mui/icons-material';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../services/AuthContext';
import { useUIState, uiActions } from '../contexts/UIStateContext';
import { useProject } from '../contexts/ProjectContext';
import projectService from '../services/ProjectService';
import MaterialLibrary from './dialogs/MaterialLibrary';
import MeshGeneration from './dialogs/MeshGeneration';
import BoundaryConditions from './dialogs/BoundaryConditions';
import MeshVisualization from './dialogs/MeshVisualization';
import MaterialsAssignment from './dialogs/MaterialsAssignment';
import ProjectCreation from './dialogs/ProjectCreation';
import ProjectManager from './dialogs/ProjectManager';
import { sampleMeshData } from '../data/sampleMeshData';

const ProfessionalNavigation: React.FC = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { dispatch } = useUIState();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [fileMenuAnchor, setFileMenuAnchor] = useState<null | HTMLElement>(null);
  const [viewMenuAnchor, setViewMenuAnchor] = useState<null | HTMLElement>(null);
  const [toolsMenuAnchor, setToolsMenuAnchor] = useState<null | HTMLElement>(null);
  const [analysisMenuAnchor, setAnalysisMenuAnchor] = useState<null | HTMLElement>(null);
  const [resultsMenuAnchor, setResultsMenuAnchor] = useState<null | HTMLElement>(null);
  const [isAnalysisRunning, setIsAnalysisRunning] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showGrid, setShowGrid] = useState(true);
  const [showModelTree, setShowModelTree] = useState(true);

  // Dialog states
  const [materialLibraryOpen, setMaterialLibraryOpen] = useState(false);
  const [meshGenerationOpen, setMeshGenerationOpen] = useState(false);
  const [boundaryConditionsOpen, setBoundaryConditionsOpen] = useState(false);
  const [meshVisualizationOpen, setMeshVisualizationOpen] = useState(false);
  const [materialsAssignmentOpen, setMaterialsAssignmentOpen] = useState(false);

  const handleUserMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleCloseUserMenu = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
    handleCloseUserMenu();
  };

  const handleFileMenu = (event: React.MouseEvent<HTMLElement>) => {
    setFileMenuAnchor(event.currentTarget);
  };

  const handleViewMenu = (event: React.MouseEvent<HTMLElement>) => {
    setViewMenuAnchor(event.currentTarget);
  };

  const handleToolsMenu = (event: React.MouseEvent<HTMLElement>) => {
    setToolsMenuAnchor(event.currentTarget);
  };

  const handleAnalysisMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnalysisMenuAnchor(event.currentTarget);
  };

  const handleResultsMenu = (event: React.MouseEvent<HTMLElement>) => {
    setResultsMenuAnchor(event.currentTarget);
  };

  const handleSaveProject = async () => {
    try {
      console.log('Saving project...');
      const currentProject = projectService.getCurrentProject();

      if (currentProject) {
        await projectService.saveProject();
        console.log('Project saved successfully');
      } else {
        console.log('No active project to save');
      }
    } catch (error) {
      console.error('Failed to save project:', error);
      // TODO: Show error dialog
    }
  };

  const handleOpenProject = async () => {
    try {
      console.log('Opening project...');
      const project = await projectService.openProject();

      if (project) {
        console.log('Project opened successfully:', project.name);
        // TODO: Update UI state to reflect new project
      } else {
        console.log('No project was opened');
      }
    } catch (error) {
      console.error('Failed to open project:', error);
      // TODO: Show error dialog
    }
  };

  const handleNewProject = async () => {
    try {
      console.log('Creating new project...');
      const projectName = prompt('Enter project name:', 'New Project');

      if (projectName) {
        const project = await projectService.createNewProject(projectName);
        console.log('New project created:', project.name);
        // TODO: Update UI state to reflect new project
      }
    } catch (error) {
      console.error('Failed to create new project:', error);
      // TODO: Show error dialog
    }
  };

  const handleSaveAsProject = async () => {
    try {
      console.log('Save As project...');
      const filePath = await projectService.saveAsProject();
      console.log('Project saved as:', filePath);
    } catch (error) {
      console.error('Failed to save project as:', error);
      // TODO: Show error dialog
    }
  };

  const handleExportProject = async (format: 'json' | 'xml' | 'step' | 'iges' | 'stl') => {
    try {
      console.log(`Exporting project as ${format}...`);
      await projectService.exportProject(format);
      console.log(`Project exported as ${format} successfully`);
    } catch (error) {
      console.error(`Failed to export project as ${format}:`, error);
      // TODO: Show error dialog
    }
  };

  const handlePrintReport = async () => {
    try {
      console.log('Generating report...');
      const currentProject = projectService.getCurrentProject();

      if (currentProject) {
        // TODO: Generate and print comprehensive report
        console.log('Report generated for project:', currentProject.name);
        window.print(); // Temporary implementation
      } else {
        console.log('No active project to report on');
      }
    } catch (error) {
      console.error('Failed to generate report:', error);
      // TODO: Show error dialog
    }
  };

  const handleRefresh = () => {
    console.log('Refreshing...');
    window.location.reload();
  };

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.2, 5));
    console.log('Zoom In:', zoomLevel * 1.2);
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.1));
    console.log('Zoom Out:', zoomLevel / 1.2);
  };

  const handleFitToWindow = () => {
    setZoomLevel(1);
    console.log('Fit to Window - Reset zoom to 1');
  };

  const handleRunAnalysis = () => {
    setIsAnalysisRunning(true);
    console.log('Starting analysis...');
    // Simulate analysis completion after 5 seconds
    setTimeout(() => {
      setIsAnalysisRunning(false);
      console.log('Analysis completed');
    }, 5000);
  };

  const handlePauseAnalysis = () => {
    console.log('Pausing analysis...');
    // TODO: Implement actual pause functionality
  };

  const handleStopAnalysis = () => {
    setIsAnalysisRunning(false);
    console.log('Stopping analysis...');
  };

  const handleToggleGrid = () => {
    setShowGrid(prev => !prev);
    console.log('Grid visibility:', !showGrid);
  };

  const handleToggleModelTree = () => {
    setShowModelTree(prev => !prev);
    console.log('Model tree visibility:', !showModelTree);
  };

  // Tools Menu Handlers
  const handleMeshGeneration = () => {
    setMeshGenerationOpen(true);
  };

  const handleMaterialLibrary = () => {
    setMaterialLibraryOpen(true);
  };

  const handleBoundaryConditions = () => {
    setBoundaryConditionsOpen(true);
  };

  const handleMeshVisualization = () => {
    setMeshVisualizationOpen(true);
  };

  const handleMaterialsAssignment = () => {
    setMaterialsAssignmentOpen(true);
  };

  const handleModelValidation = () => {
    console.log('Running model validation...');
    // TODO: Perform comprehensive model validation:
    // - Geometry integrity check
    // - Mesh quality assessment
    // - Material property verification
    // - Boundary condition completeness
  };

  const handleUnitsManager = () => {
    console.log('Opening units manager...');
    // TODO: Open units management dialog for:
    // - Base unit system selection
    // - Unit conversion tools
    // - Dimensional analysis
  };

  const handleParametricStudy = () => {
    console.log('Setting up parametric study...');
    // TODO: Open parametric study configuration:
    // - Parameter definition and ranges
    // - Design of experiments (DOE)
    // - Optimization algorithms
  };

  // Analysis Menu Handlers
  const handleStaticAnalysis = () => {
    dispatch(uiActions.setAnalysisRunning(true));
    console.log('Starting static analysis...');
    // TODO: Configure and run static analysis:
    // - Linear/nonlinear solution options
    // - Convergence criteria
    // - Solution monitoring
    setTimeout(() => {
      dispatch(uiActions.setAnalysisRunning(false));
      console.log('Static analysis completed');
    }, 5000);
  };

  const handleModalAnalysis = () => {
    dispatch(uiActions.setAnalysisRunning(true));
    console.log('Starting modal analysis...');
    // TODO: Configure modal analysis:
    // - Number of modes to extract
    // - Frequency range
    // - Mode normalization options
    setTimeout(() => {
      dispatch(uiActions.setAnalysisRunning(false));
      console.log('Modal analysis completed');
    }, 4000);
  };

  const handleHarmonicAnalysis = () => {
    dispatch(uiActions.setAnalysisRunning(true));
    console.log('Starting harmonic analysis...');
    // TODO: Configure harmonic analysis:
    // - Frequency sweep parameters
    // - Damping specification
    // - Response calculation options
    setTimeout(() => {
      dispatch(uiActions.setAnalysisRunning(false));
      console.log('Harmonic analysis completed');
    }, 6000);
  };

  const handleTransientAnalysis = () => {
    dispatch(uiActions.setAnalysisRunning(true));
    console.log('Starting transient analysis...');
    // TODO: Configure transient analysis:
    // - Time integration parameters
    // - Time step control
    // - Solution stability monitoring
    setTimeout(() => {
      dispatch(uiActions.setAnalysisRunning(false));
      console.log('Transient analysis completed');
    }, 8000);
  };

  const handleNonlinearAnalysis = () => {
    dispatch(uiActions.setAnalysisRunning(true));
    console.log('Starting nonlinear analysis...');
    // TODO: Configure nonlinear analysis:
    // - Material nonlinearity (plasticity, hyperelasticity)
    // - Geometric nonlinearity (large deformations)
    // - Contact nonlinearity
    setTimeout(() => {
      dispatch(uiActions.setAnalysisRunning(false));
      console.log('Nonlinear analysis completed');
    }, 10000);
  };

  const handleThermalAnalysis = () => {
    dispatch(uiActions.setAnalysisRunning(true));
    console.log('Starting thermal analysis...');
    // TODO: Configure thermal analysis:
    // - Heat transfer mechanisms (conduction, convection, radiation)
    // - Temperature-dependent properties
    // - Thermal boundary conditions
    setTimeout(() => {
      dispatch(uiActions.setAnalysisRunning(false));
      console.log('Thermal analysis completed');
    }, 7000);
  };

  const handleSolverSettings = () => {
    console.log('Opening solver settings...');
    // TODO: Open solver configuration dialog:
    // - Solver selection (direct, iterative)
    // - Convergence tolerances
    // - Memory and performance optimization
    // - Parallel processing options
  };

  // Results Menu Handlers
  const handleViewDeformation = () => {
    console.log('Displaying deformation results...');
    // TODO: Configure deformation visualization:
    // - Deformation scaling options
    // - Animation controls
    // - Undeformed/deformed overlay
    dispatch({ type: 'SET_SELECTED_VISUALIZATION', payload: 0 });
  };

  const handleViewStress = () => {
    console.log('Displaying stress results...');
    // TODO: Configure stress visualization:
    // - Stress component selection (von Mises, principal, etc.)
    // - Contour plot options
    // - Critical stress identification
    dispatch({ type: 'SET_SELECTED_VISUALIZATION', payload: 1 });
  };

  const handleViewStrain = () => {
    console.log('Displaying strain results...');
    // TODO: Configure strain visualization:
    // - Strain component selection
    // - Strain rosette analysis
    // - Local coordinate system definition
    dispatch({ type: 'SET_SELECTED_VISUALIZATION', payload: 2 });
  };

  const handleSafetyFactor = () => {
    console.log('Calculating safety factors...');
    // TODO: Configure safety factor analysis:
    // - Failure criteria selection
    // - Material strength properties
    // - Safety margin assessment
  };

  const handleCreateAnimation = () => {
    console.log('Creating result animation...');
    dispatch(uiActions.toggleAnimation(true));
    // TODO: Configure animation creation:
    // - Animation sequence definition
    // - Frame rate and duration controls
    // - Export format options (AVI, GIF, MP4)
  };

  const handleResultsComparison = () => {
    console.log('Opening results comparison...');
    // TODO: Configure results comparison:
    // - Multiple load case comparison
    // - Design iteration comparison
    // - Benchmark validation
  };

  const handleExportResults = () => {
    console.log('Exporting results...');
    // TODO: Configure results export:
    // - Data format selection (CSV, Excel, HDF5)
    // - Result quantity selection
    // - Coordinate system options
  };

  const handleGenerateReport = () => {
    console.log('Generating analysis report...');
    // TODO: Configure report generation:
    // - Report template selection
    // - Content customization
    // - Export format (PDF, Word, HTML)
  };

  const isActiveRoute = (path: string) => {
    return location.pathname === path;
  };

  return (
    <>
      {/* Main Menu Bar */}
      <AppBar position="static" sx={{
        backgroundColor: '#1e1e1e',
        borderBottom: '1px solid #3e3e42',
        zIndex: 1300
      }}>
        <Toolbar sx={{ minHeight: '32px !important', padding: '0 8px' }}>
          {/* Application Title */}
          <Typography
            variant="h6"
            sx={{
              fontSize: '14px',
              fontWeight: 600,
              color: '#ffffff',
              mr: 3
            }}
          >
            Engineering Analysis Workbench
          </Typography>

          {/* Main Menu Items */}
          {isAuthenticated && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Button
                color="inherit"
                size="small"
                onClick={handleFileMenu}
                sx={{
                  fontSize: '12px',
                  minWidth: 'auto',
                  padding: '4px 8px',
                  '&:hover': { backgroundColor: '#3e3e42' }
                }}
              >
                File
              </Button>

              <Button
                color="inherit"
                size="small"
                onClick={handleViewMenu}
                sx={{
                  fontSize: '12px',
                  minWidth: 'auto',
                  padding: '4px 8px',
                  '&:hover': { backgroundColor: '#3e3e42' }
                }}
              >
                View
              </Button>

              <Button
                color="inherit"
                size="small"
                onClick={handleToolsMenu}
                sx={{
                  fontSize: '12px',
                  minWidth: 'auto',
                  padding: '4px 8px',
                  '&:hover': { backgroundColor: '#3e3e42' }
                }}
              >
                Tools
              </Button>

              <Button
                color="inherit"
                size="small"
                onClick={handleAnalysisMenu}
                sx={{
                  fontSize: '12px',
                  minWidth: 'auto',
                  padding: '4px 8px',
                  '&:hover': { backgroundColor: '#3e3e42' }
                }}
              >
                Analysis
              </Button>

              <Button
                color="inherit"
                size="small"
                onClick={handleResultsMenu}
                sx={{
                  fontSize: '12px',
                  minWidth: 'auto',
                  padding: '4px 8px',
                  '&:hover': { backgroundColor: '#3e3e42' }
                }}
              >
                Results
              </Button>
            </Box>
          )}

          <Box sx={{ flexGrow: 1 }} />

          {/* Status Indicators */}
          {isAuthenticated && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mr: 2 }}>
              <Chip
                label="Solver: Ready"
                size="small"
                sx={{
                  height: '20px',
                  fontSize: '10px',
                  backgroundColor: '#28a745',
                  color: 'white'
                }}
              />
              <Chip
                label="Units: SI (m)"
                size="small"
                sx={{
                  height: '20px',
                  fontSize: '10px',
                  backgroundColor: '#2d2d30',
                  color: '#b0b0b0',
                  border: '1px solid #3e3e42'
                }}
              />
            </Box>
          )}

          {/* User Menu */}
          {isAuthenticated && (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <IconButton
                size="small"
                onClick={handleUserMenu}
                color="inherit"
                sx={{ padding: '4px' }}
              >
                <AccountCircle sx={{ fontSize: '18px' }} />
              </IconButton>
            </Box>
          )}
        </Toolbar>
      </AppBar>

      {/* Professional Ribbon/Toolbar */}
      {isAuthenticated && (
        <Paper
          elevation={0}
          sx={{
            backgroundColor: '#2d2d30',
            borderBottom: '1px solid #3e3e42',
            padding: '4px 8px'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* Quick Access Toolbar */}
            <ButtonGroup size="small" variant="outlined">
              <Tooltip title="Save Project">
                <IconButton size="small" sx={{ border: '1px solid #3e3e42' }} onClick={handleSaveProject}>
                  <Save sx={{ fontSize: '16px' }} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Open Project">
                <IconButton size="small" sx={{ border: '1px solid #3e3e42' }} onClick={handleOpenProject}>
                  <FolderOpen sx={{ fontSize: '16px' }} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Refresh">
                <IconButton size="small" sx={{ border: '1px solid #3e3e42' }} onClick={handleRefresh}>
                  <Refresh sx={{ fontSize: '16px' }} />
                </IconButton>
              </Tooltip>
            </ButtonGroup>

            <Divider orientation="vertical" flexItem sx={{ backgroundColor: '#3e3e42' }} />

            {/* Navigation Buttons */}
            <ButtonGroup size="small" variant="outlined">
              <Tooltip title="Dashboard">
                <Button
                  component={Link}
                  to="/"
                  variant={isActiveRoute('/') ? 'contained' : 'outlined'}
                  startIcon={<Analytics sx={{ fontSize: '16px' }} />}
                  sx={{
                    fontSize: '11px',
                    minWidth: 'auto',
                    padding: '4px 8px',
                    backgroundColor: isActiveRoute('/') ? '#00a8ff' : 'transparent'
                  }}
                >
                  Dashboard
                </Button>
              </Tooltip>
              <Tooltip title="Analysis Studio">
                <Button
                  component={Link}
                  to="/visualizations"
                  variant={isActiveRoute('/visualizations') ? 'contained' : 'outlined'}
                  startIcon={<ViewInAr sx={{ fontSize: '16px' }} />}
                  sx={{
                    fontSize: '11px',
                    minWidth: 'auto',
                    padding: '4px 8px',
                    backgroundColor: isActiveRoute('/visualizations') ? '#00a8ff' : 'transparent'
                  }}
                >
                  Analysis
                </Button>
              </Tooltip>
              <Tooltip title="Model Setup">
                <Button
                  component={Link}
                  to="/models"
                  variant={isActiveRoute('/models') ? 'contained' : 'outlined'}
                  startIcon={<Engineering sx={{ fontSize: '16px' }} />}
                  sx={{
                    fontSize: '11px',
                    minWidth: 'auto',
                    padding: '4px 8px',
                    backgroundColor: isActiveRoute('/models') ? '#00a8ff' : 'transparent'
                  }}
                >
                  Models
                </Button>
              </Tooltip>
              <Tooltip title="Next-Gen ML">
                <Button
                  component={Link}
                  to="/nextgen-ml"
                  variant={isActiveRoute('/nextgen-ml') ? 'contained' : 'outlined'}
                  startIcon={<Science sx={{ fontSize: '16px' }} />}
                  sx={{
                    fontSize: '11px',
                    minWidth: 'auto',
                    padding: '4px 8px',
                    backgroundColor: isActiveRoute('/nextgen-ml') ? '#00a8ff' : 'transparent'
                  }}
                >
                  AI/ML
                </Button>
              </Tooltip>
            </ButtonGroup>

            <Divider orientation="vertical" flexItem sx={{ backgroundColor: '#3e3e42' }} />

            {/* View Controls */}
            <ButtonGroup size="small" variant="outlined">
              <Tooltip title={`Zoom In (${Math.round(zoomLevel * 120)}%)`}>
                <IconButton size="small" sx={{ border: '1px solid #3e3e42' }} onClick={handleZoomIn}>
                  <ZoomIn sx={{ fontSize: '16px' }} />
                </IconButton>
              </Tooltip>
              <Tooltip title={`Zoom Out (${Math.round(zoomLevel / 1.2 * 100)}%)`}>
                <IconButton size="small" sx={{ border: '1px solid #3e3e42' }} onClick={handleZoomOut}>
                  <ZoomOut sx={{ fontSize: '16px' }} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Fit to Window (100%)">
                <IconButton size="small" sx={{ border: '1px solid #3e3e42' }} onClick={handleFitToWindow}>
                  <CenterFocusStrong sx={{ fontSize: '16px' }} />
                </IconButton>
              </Tooltip>
            </ButtonGroup>

            <Divider orientation="vertical" flexItem sx={{ backgroundColor: '#3e3e42' }} />

            {/* Simulation Controls */}
            <ButtonGroup size="small" variant="outlined">
              <Tooltip title={isAnalysisRunning ? "Analysis Running..." : "Run Analysis"}>
                <IconButton
                  size="small"
                  sx={{
                    border: '1px solid #3e3e42',
                    color: isAnalysisRunning ? '#ff9800' : '#28a745',
                    animation: isAnalysisRunning ? 'pulse 2s infinite' : 'none'
                  }}
                  onClick={handleRunAnalysis}
                  disabled={isAnalysisRunning}
                >
                  <PlayArrow sx={{ fontSize: '16px' }} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Pause Analysis">
                <IconButton
                  size="small"
                  sx={{ border: '1px solid #3e3e42' }}
                  onClick={handlePauseAnalysis}
                  disabled={!isAnalysisRunning}
                >
                  <Pause sx={{ fontSize: '16px' }} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Stop Analysis">
                <IconButton
                  size="small"
                  sx={{ border: '1px solid #3e3e42', color: '#dc3545' }}
                  onClick={handleStopAnalysis}
                  disabled={!isAnalysisRunning}
                >
                  <Stop sx={{ fontSize: '16px' }} />
                </IconButton>
              </Tooltip>
            </ButtonGroup>

            <Box sx={{ flexGrow: 1 }} />

            {/* System Status */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Badge badgeContent={3} color="error">
                <BugReport sx={{ fontSize: '16px', color: '#b0b0b0' }} />
              </Badge>
              <Badge badgeContent={12} color="primary">
                <Memory sx={{ fontSize: '16px', color: '#b0b0b0' }} />
              </Badge>
              <Speed sx={{ fontSize: '16px', color: '#28a745' }} />
            </Box>
          </Box>
        </Paper>
      )}

      {/* File Menu */}
      <Menu
        anchorEl={fileMenuAnchor}
        open={Boolean(fileMenuAnchor)}
        onClose={() => setFileMenuAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#2d2d30',
            border: '1px solid #3e3e42',
            minWidth: '200px'
          }
        }}
      >
        <MenuItem onClick={() => { handleNewProject(); setFileMenuAnchor(null); }}>
          <Add sx={{ mr: 1, fontSize: '16px' }} />
          New Project
        </MenuItem>
        <MenuItem onClick={() => { handleOpenProject(); setFileMenuAnchor(null); }}>
          <FolderOpen sx={{ mr: 1, fontSize: '16px' }} />
          Open Project
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={() => { handleSaveProject(); setFileMenuAnchor(null); }}>
          <Save sx={{ mr: 1, fontSize: '16px' }} />
          Save Project
        </MenuItem>
        <MenuItem onClick={() => { handleSaveAsProject(); setFileMenuAnchor(null); }}>
          <Save sx={{ mr: 1, fontSize: '16px' }} />
          Save As...
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={() => { handleExportProject('json'); setFileMenuAnchor(null); }}>
          <CloudDownload sx={{ mr: 1, fontSize: '16px' }} />
          Export Project (JSON)
        </MenuItem>
        <MenuItem onClick={() => { handleExportProject('step'); setFileMenuAnchor(null); }}>
          <CloudDownload sx={{ mr: 1, fontSize: '16px' }} />
          Export Geometry (STEP)
        </MenuItem>
        <MenuItem onClick={() => { handleExportProject('stl'); setFileMenuAnchor(null); }}>
          <CloudDownload sx={{ mr: 1, fontSize: '16px' }} />
          Export Mesh (STL)
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={() => { handlePrintReport(); setFileMenuAnchor(null); }}>
          <Print sx={{ mr: 1, fontSize: '16px' }} />
          Generate Report
        </MenuItem>
      </Menu>

      {/* View Menu */}
      <Menu
        anchorEl={viewMenuAnchor}
        open={Boolean(viewMenuAnchor)}
        onClose={() => setViewMenuAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#2d2d30',
            border: '1px solid #3e3e42',
            minWidth: '200px'
          }
        }}
      >
        <MenuItem onClick={() => { handleToggleGrid(); setViewMenuAnchor(null); }}>
          <GridOn sx={{ mr: 1, fontSize: '16px', color: showGrid ? '#00a8ff' : 'inherit' }} />
          {showGrid ? 'Hide Grid' : 'Show Grid'}
        </MenuItem>
        <MenuItem onClick={() => setViewMenuAnchor(null)}>
          <Layers sx={{ mr: 1, fontSize: '16px' }} />
          Layers Panel
        </MenuItem>
        <MenuItem onClick={() => { handleToggleModelTree(); setViewMenuAnchor(null); }}>
          <Visibility sx={{ mr: 1, fontSize: '16px', color: showModelTree ? '#00a8ff' : 'inherit' }} />
          {showModelTree ? 'Hide Model Tree' : 'Show Model Tree'}
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={() => setViewMenuAnchor(null)}>
          <CenterFocusStrong sx={{ mr: 1, fontSize: '16px' }} />
          Reset View
        </MenuItem>
      </Menu>

      {/* User Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleCloseUserMenu}
        PaperProps={{
          sx: {
            backgroundColor: '#2d2d30',
            border: '1px solid #3e3e42',
            minWidth: '200px'
          }
        }}
      >
        <MenuItem disabled sx={{ fontSize: '12px', opacity: 0.7 }}>
          {user?.email}
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={handleCloseUserMenu}>
          <Settings sx={{ mr: 1, fontSize: '16px' }} />
          Preferences
        </MenuItem>
        <MenuItem onClick={handleCloseUserMenu}>
          <Help sx={{ mr: 1, fontSize: '16px' }} />
          Help & Support
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={handleLogout}>
          <ExitToApp sx={{ mr: 1, fontSize: '16px' }} />
          Sign Out
        </MenuItem>
      </Menu>

      {/* Tools Menu */}
      <Menu
        anchorEl={toolsMenuAnchor}
        open={Boolean(toolsMenuAnchor)}
        onClose={() => setToolsMenuAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#2d2d30',
            border: '1px solid #3e3e42',
            minWidth: '250px'
          }
        }}
      >
        <MenuItem onClick={() => { handleMeshGeneration(); setToolsMenuAnchor(null); }}>
          <Build sx={{ mr: 1, fontSize: '16px' }} />
          Mesh Generation
        </MenuItem>
        <MenuItem onClick={() => { handleMaterialLibrary(); setToolsMenuAnchor(null); }}>
          <Memory sx={{ mr: 1, fontSize: '16px' }} />
          Material Library
        </MenuItem>
        <MenuItem onClick={() => { handleBoundaryConditions(); setToolsMenuAnchor(null); }}>
          <Engineering sx={{ mr: 1, fontSize: '16px' }} />
          Boundary Conditions
        </MenuItem>
        <MenuItem onClick={() => { handleMeshVisualization(); setToolsMenuAnchor(null); }}>
          <Speed sx={{ mr: 1, fontSize: '16px' }} />
          Mesh Visualization
        </MenuItem>
        <MenuItem onClick={() => { handleMaterialsAssignment(); setToolsMenuAnchor(null); }}>
          <Science sx={{ mr: 1, fontSize: '16px' }} />
          Materials Assignment
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={() => { handleModelValidation(); setToolsMenuAnchor(null); }}>
          <Analytics sx={{ mr: 1, fontSize: '16px' }} />
          Model Validation
        </MenuItem>
        <MenuItem onClick={() => { handleUnitsManager(); setToolsMenuAnchor(null); }}>
          <Settings sx={{ mr: 1, fontSize: '16px' }} />
          Units Manager
        </MenuItem>
        <MenuItem onClick={() => { handleParametricStudy(); setToolsMenuAnchor(null); }}>
          <Timeline sx={{ mr: 1, fontSize: '16px' }} />
          Parametric Study
        </MenuItem>
      </Menu>

      {/* Analysis Menu */}
      <Menu
        anchorEl={analysisMenuAnchor}
        open={Boolean(analysisMenuAnchor)}
        onClose={() => setAnalysisMenuAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#2d2d30',
            border: '1px solid #3e3e42',
            minWidth: '250px'
          }
        }}
      >
        <MenuItem onClick={() => { handleStaticAnalysis(); setAnalysisMenuAnchor(null); }}>
          <PlayArrow sx={{ mr: 1, fontSize: '16px', color: '#28a745' }} />
          Static Analysis
        </MenuItem>
        <MenuItem onClick={() => { handleModalAnalysis(); setAnalysisMenuAnchor(null); }}>
          <Timeline sx={{ mr: 1, fontSize: '16px' }} />
          Modal Analysis
        </MenuItem>
        <MenuItem onClick={() => { handleHarmonicAnalysis(); setAnalysisMenuAnchor(null); }}>
          <TrendingUp sx={{ mr: 1, fontSize: '16px' }} />
          Harmonic Analysis
        </MenuItem>
        <MenuItem onClick={() => { handleTransientAnalysis(); setAnalysisMenuAnchor(null); }}>
          <Speed sx={{ mr: 1, fontSize: '16px' }} />
          Transient Analysis
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={() => { handleNonlinearAnalysis(); setAnalysisMenuAnchor(null); }}>
          <Analytics sx={{ mr: 1, fontSize: '16px', color: '#ff9800' }} />
          Nonlinear Analysis
        </MenuItem>
        <MenuItem onClick={() => { handleThermalAnalysis(); setAnalysisMenuAnchor(null); }}>
          <Memory sx={{ mr: 1, fontSize: '16px', color: '#f44336' }} />
          Thermal Analysis
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={() => { handleSolverSettings(); setAnalysisMenuAnchor(null); }}>
          <Settings sx={{ mr: 1, fontSize: '16px' }} />
          Solver Settings
        </MenuItem>
      </Menu>

      {/* Results Menu */}
      <Menu
        anchorEl={resultsMenuAnchor}
        open={Boolean(resultsMenuAnchor)}
        onClose={() => setResultsMenuAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#2d2d30',
            border: '1px solid #3e3e42',
            minWidth: '250px'
          }
        }}
      >
        <MenuItem onClick={() => { handleViewDeformation(); setResultsMenuAnchor(null); }}>
          <Visibility sx={{ mr: 1, fontSize: '16px' }} />
          View Deformation
        </MenuItem>
        <MenuItem onClick={() => { handleViewStress(); setResultsMenuAnchor(null); }}>
          <TrendingUp sx={{ mr: 1, fontSize: '16px' }} />
          Stress Analysis
        </MenuItem>
        <MenuItem onClick={() => { handleViewStrain(); setResultsMenuAnchor(null); }}>
          <Timeline sx={{ mr: 1, fontSize: '16px' }} />
          Strain Analysis
        </MenuItem>
        <MenuItem onClick={() => { handleSafetyFactor(); setResultsMenuAnchor(null); }}>
          <Analytics sx={{ mr: 1, fontSize: '16px', color: '#28a745' }} />
          Safety Factor
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={() => { handleCreateAnimation(); setResultsMenuAnchor(null); }}>
          <PlayArrow sx={{ mr: 1, fontSize: '16px' }} />
          Create Animation
        </MenuItem>
        <MenuItem onClick={() => { handleResultsComparison(); setResultsMenuAnchor(null); }}>
          <Analytics sx={{ mr: 1, fontSize: '16px' }} />
          Results Comparison
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={() => { handleExportResults(); setResultsMenuAnchor(null); }}>
          <CloudDownload sx={{ mr: 1, fontSize: '16px' }} />
          Export Results
        </MenuItem>
        <MenuItem onClick={() => { handleGenerateReport(); setResultsMenuAnchor(null); }}>
          <Print sx={{ mr: 1, fontSize: '16px' }} />
          Generate Report
        </MenuItem>
      </Menu>

      {/* Dialogs */}
      <MaterialLibrary
        open={materialLibraryOpen}
        onClose={() => setMaterialLibraryOpen(false)}
        onMaterialSelect={(material) => {
          console.log('Material selected:', material.name);
          // TODO: Apply material to selected geometry
        }}
      />

      <MeshGeneration
        open={meshGenerationOpen}
        onClose={() => setMeshGenerationOpen(false)}
        onGenerate={(meshConfig) => {
          console.log('Generating mesh with config:', meshConfig);
          dispatch(uiActions.setAnalysisRunning(true));
          setTimeout(() => {
            dispatch(uiActions.setAnalysisRunning(false));
            console.log('Mesh generation completed');
          }, 3000);
        }}
      />

      <BoundaryConditions
        open={boundaryConditionsOpen}
        onClose={() => setBoundaryConditionsOpen(false)}
        onApply={(conditions) => {
          console.log('Applying boundary conditions:', conditions);
          // TODO: Apply boundary conditions to model
        }}
      />

      <MeshVisualization
        open={meshVisualizationOpen}
        onClose={() => setMeshVisualizationOpen(false)}
        meshData={sampleMeshData}
      />

      <MaterialsAssignment
        open={materialsAssignmentOpen}
        onClose={() => setMaterialsAssignmentOpen(false)}
        onAssignMaterial={(bodyId, materialId) => {
          console.log('Assigning material:', { bodyId, materialId });
          // TODO: Apply material assignment to geometry body
        }}
      />
    </>
  );
};

export default ProfessionalNavigation;