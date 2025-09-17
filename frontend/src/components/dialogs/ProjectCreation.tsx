import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  CardActions,
  Alert
} from '@mui/material';
import {
  Engineering,
  Whatshot,
  WaterDrop,
  Science,
  Add
} from '@mui/icons-material';
import { useProject } from '../../contexts/ProjectContext';
import { useAuth } from '../../services/AuthContext';

interface ProjectCreationProps {
  open: boolean;
  onClose: () => void;
}

export default function ProjectCreation({ open, onClose }: ProjectCreationProps) {
  const [projectName, setProjectName] = useState('');
  const [description, setDescription] = useState('');
  const [projectType, setProjectType] = useState<'structural' | 'thermal' | 'fluid' | 'multiphysics'>('structural');
  const [error, setError] = useState('');
  const { createProject } = useProject();
  const { user } = useAuth();

  const projectTypes = [
    {
      id: 'structural',
      name: 'Structural Analysis',
      description: 'Stress, strain, deformation, and fatigue analysis',
      icon: <Engineering sx={{ fontSize: 40 }} />,
      color: '#1976d2'
    },
    {
      id: 'thermal',
      name: 'Thermal Analysis',
      description: 'Heat transfer, temperature distribution, thermal stress',
      icon: <Whatshot sx={{ fontSize: 40 }} />,
      color: '#d32f2f'
    },
    {
      id: 'fluid',
      name: 'Fluid Dynamics',
      description: 'CFD, flow analysis, pressure distribution',
      icon: <WaterDrop sx={{ fontSize: 40 }} />,
      color: '#1976d2'
    },
    {
      id: 'multiphysics',
      name: 'Multiphysics',
      description: 'Coupled physics simulations',
      icon: <Science sx={{ fontSize: 40 }} />,
      color: '#9c27b0'
    }
  ];

  const handleCreateProject = () => {
    if (!projectName.trim()) {
      setError('Project name is required');
      return;
    }

    try {
      createProject({
        name: projectName.trim(),
        description: description.trim(),
        type: projectType,
        author: user?.full_name || 'Unknown',
        meshData: null,
        materials: [],
        boundaryConditions: [],
        results: [],
        settings: {
          units: {
            length: 'mm',
            force: 'N',
            pressure: 'MPa',
            temperature: 'C'
          },
          solver: 'default',
          meshQuality: 'medium'
        }
      });

      // Reset form
      setProjectName('');
      setDescription('');
      setProjectType('structural');
      setError('');
      onClose();
    } catch (err) {
      setError('Failed to create project');
    }
  };

  const handleTemplateSelect = (templateType: string) => {
    setProjectType(templateType as any);

    switch (templateType) {
      case 'structural':
        setProjectName('Structural Analysis Project');
        setDescription('New structural analysis project for stress and deformation analysis');
        break;
      case 'thermal':
        setProjectName('Thermal Analysis Project');
        setDescription('New thermal analysis project for heat transfer simulation');
        break;
      case 'fluid':
        setProjectName('CFD Project');
        setDescription('New computational fluid dynamics project');
        break;
      case 'multiphysics':
        setProjectName('Multiphysics Project');
        setDescription('New coupled physics simulation project');
        break;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { minHeight: '600px' }
      }}
    >
      <DialogTitle sx={{ pb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Add />
          Create New Project
        </Box>
      </DialogTitle>

      <DialogContent>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Choose Project Template
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Select the type of analysis you want to perform
          </Typography>

          <Grid container spacing={2}>
            {projectTypes.map((type) => (
              <Grid item xs={12} sm={6} md={3} key={type.id}>
                <Card
                  sx={{
                    height: '100%',
                    cursor: 'pointer',
                    border: projectType === type.id ? 2 : 1,
                    borderColor: projectType === type.id ? type.color : 'divider',
                    '&:hover': {
                      borderColor: type.color,
                      elevation: 4
                    }
                  }}
                  onClick={() => handleTemplateSelect(type.id)}
                >
                  <CardContent sx={{ textAlign: 'center', pb: 1 }}>
                    <Box sx={{ color: type.color, mb: 1 }}>
                      {type.icon}
                    </Box>
                    <Typography variant="h6" gutterBottom>
                      {type.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {type.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Project Details
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Project Name"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="Enter project name"
                required
              />
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Enter project description (optional)"
                multiline
                rows={3}
              />
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Analysis Type</InputLabel>
                <Select
                  value={projectType}
                  label="Analysis Type"
                  onChange={(e) => setProjectType(e.target.value as any)}
                >
                  {projectTypes.map((type) => (
                    <MenuItem key={type.id} value={type.id}>
                      {type.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Author"
                value={user?.full_name || ''}
                disabled
              />
            </Grid>
          </Grid>
        </Box>

        <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
          <Typography variant="body2" color="text.secondary">
            <strong>What's included:</strong>
            <br />
            • Pre-configured analysis settings
            <br />
            • Standard material library
            <br />
            • Default meshing parameters
            <br />
            • Common boundary condition templates
          </Typography>
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 3, pt: 0 }}>
        <Button onClick={onClose} color="inherit">
          Cancel
        </Button>
        <Button
          onClick={handleCreateProject}
          variant="contained"
          disabled={!projectName.trim()}
        >
          Create Project
        </Button>
      </DialogActions>
    </Dialog>
  );
}