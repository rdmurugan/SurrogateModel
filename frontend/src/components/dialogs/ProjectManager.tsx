import React, { useState, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  CardActions,
  IconButton,
  Menu,
  MenuItem,
  Alert,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction
} from '@mui/material';
import {
  FolderOpen,
  Save,
  Download,
  Upload,
  Delete,
  MoreVert,
  Engineering,
  Whatshot,
  WaterDrop,
  Science,
  FileUpload,
  GetApp
} from '@mui/icons-material';
import { useProject, Project } from '../../contexts/ProjectContext';

interface ProjectManagerProps {
  open: boolean;
  onClose: () => void;
  mode: 'open' | 'save' | 'export' | 'import';
}

export default function ProjectManager({ open, onClose, mode }: ProjectManagerProps) {
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [menuProjectId, setMenuProjectId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [exportFormat, setExportFormat] = useState('json');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { projects, currentProject, openProject, deleteProject, exportProject, importProject } = useProject();

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'structural': return <Engineering />;
      case 'thermal': return <Whatshot />;
      case 'fluid': return <WaterDrop />;
      case 'multiphysics': return <Science />;
      default: return <Engineering />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'structural': return '#1976d2';
      case 'thermal': return '#d32f2f';
      case 'fluid': return '#0288d1';
      case 'multiphysics': return '#9c27b0';
      default: return '#1976d2';
    }
  };

  const handleProjectAction = () => {
    if (mode === 'open' && selectedProject) {
      openProject(selectedProject);
      onClose();
    } else if (mode === 'save' && currentProject) {
      // Save current project (already handled by context)
      onClose();
    } else if (mode === 'export' && selectedProject) {
      exportProject(selectedProject, exportFormat);
      onClose();
    }
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>, projectId: string) => {
    setMenuAnchor(event.currentTarget);
    setMenuProjectId(projectId);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setMenuProjectId(null);
  };

  const handleDelete = () => {
    if (menuProjectId) {
      deleteProject(menuProjectId);
      handleMenuClose();
    }
  };

  const handleFileImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await importProject(file);
      onClose();
    } catch (err) {
      setError('Failed to import project: ' + err);
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const exportFormats = [
    { value: 'json', label: 'JSON (Platform Format)', ext: '.json' },
    { value: 'inp', label: 'Abaqus Input File', ext: '.inp' },
    { value: 'nas', label: 'Nastran File', ext: '.nas' },
  ];

  const renderContent = () => {
    if (mode === 'import') {
      return (
        <Box>
          <Typography variant="h6" gutterBottom>
            Import Project
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Import a project from various CAE formats
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Supported Formats
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon><FileUpload /></ListItemIcon>
                  <ListItemText
                    primary="JSON (.json)"
                    secondary="Native platform format"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><FileUpload /></ListItemIcon>
                  <ListItemText
                    primary="Abaqus Input (.inp)"
                    secondary="Abaqus finite element input files"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><FileUpload /></ListItemIcon>
                  <ListItemText
                    primary="Nastran (.nas, .bdf)"
                    secondary="MSC Nastran input files"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><FileUpload /></ListItemIcon>
                  <ListItemText
                    primary="ANSYS (.cdb)"
                    secondary="ANSYS command database files"
                  />
                </ListItem>
              </List>
            </CardContent>
            <CardActions>
              <Button
                variant="contained"
                startIcon={<Upload />}
                onClick={() => fileInputRef.current?.click()}
              >
                Choose File to Import
              </Button>
            </CardActions>
          </Card>

          <input
            ref={fileInputRef}
            type="file"
            hidden
            accept=".json,.inp,.nas,.bdf,.cdb"
            onChange={handleFileImport}
          />
        </Box>
      );
    }

    if (mode === 'export') {
      return (
        <Box>
          <Typography variant="h6" gutterBottom>
            Export Project
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Export your project to various CAE formats
          </Typography>

          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Export Format
            </Typography>
            <Grid container spacing={1}>
              {exportFormats.map((format) => (
                <Grid item key={format.value}>
                  <Chip
                    label={format.label}
                    onClick={() => setExportFormat(format.value)}
                    color={exportFormat === format.value ? 'primary' : 'default'}
                    variant={exportFormat === format.value ? 'filled' : 'outlined'}
                  />
                </Grid>
              ))}
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle1" gutterBottom>
            Select Project to Export
          </Typography>
        </Box>
      );
    }

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          {mode === 'open' ? 'Open Project' : 'Save Project'}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          {mode === 'open'
            ? 'Select a project to open and continue working'
            : 'Your current project will be saved automatically'
          }
        </Typography>

        {mode === 'save' && currentProject && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Current project "{currentProject.name}" is automatically saved.
          </Alert>
        )}
      </Box>
    );
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { minHeight: '500px' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {mode === 'open' && <FolderOpen />}
          {mode === 'save' && <Save />}
          {mode === 'export' && <Download />}
          {mode === 'import' && <Upload />}
          {mode === 'open' && 'Open Project'}
          {mode === 'save' && 'Save Project'}
          {mode === 'export' && 'Export Project'}
          {mode === 'import' && 'Import Project'}
        </Box>
      </DialogTitle>

      <DialogContent>
        {renderContent()}

        {(mode === 'open' || mode === 'export') && (
          <>
            {projects.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body1" color="text.secondary">
                  No projects found. Create a new project to get started.
                </Typography>
              </Box>
            ) : (
              <Grid container spacing={2}>
                {projects.map((project) => (
                  <Grid item xs={12} sm={6} md={4} key={project.id}>
                    <Card
                      sx={{
                        cursor: 'pointer',
                        border: selectedProject === project.id ? 2 : 1,
                        borderColor: selectedProject === project.id ? 'primary.main' : 'divider',
                        '&:hover': {
                          borderColor: 'primary.main',
                          elevation: 4
                        }
                      }}
                      onClick={() => setSelectedProject(project.id)}
                    >
                      <CardContent sx={{ pb: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 1 }}>
                          <Box sx={{ color: getTypeColor(project.type), mr: 1 }}>
                            {getTypeIcon(project.type)}
                          </Box>
                          <Box sx={{ flexGrow: 1 }}>
                            <Typography variant="h6" sx={{ fontSize: '1rem' }}>
                              {project.name}
                            </Typography>
                            <Chip
                              label={project.type}
                              size="small"
                              sx={{ fontSize: '0.7rem', height: 20 }}
                            />
                          </Box>
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleMenuClick(e, project.id);
                            }}
                          >
                            <MoreVert />
                          </IconButton>
                        </Box>

                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                          {project.description || 'No description'}
                        </Typography>

                        <Typography variant="caption" color="text.secondary">
                          Modified: {new Date(project.modified).toLocaleDateString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            )}
          </>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 3, pt: 0 }}>
        <Button onClick={onClose} color="inherit">
          Cancel
        </Button>
        {mode === 'open' && (
          <Button
            onClick={handleProjectAction}
            variant="contained"
            disabled={!selectedProject}
          >
            Open Project
          </Button>
        )}
        {mode === 'export' && (
          <Button
            onClick={handleProjectAction}
            variant="contained"
            disabled={!selectedProject}
            startIcon={<GetApp />}
          >
            Export
          </Button>
        )}
      </DialogActions>

      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleDelete}>
          <Delete sx={{ mr: 1 }} />
          Delete Project
        </MenuItem>
      </Menu>
    </Dialog>
  );
}