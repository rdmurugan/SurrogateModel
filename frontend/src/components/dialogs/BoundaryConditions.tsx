import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tabs,
  Tab,
  Box,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Typography,
  Grid,
  Card,
  CardContent,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  Alert,
  Divider,
  FormControlLabel,
  Switch,
  InputAdornment
} from '@mui/material';
import {
  Add,
  Delete,
  Edit,
  LocationOn,
  Security,
  Speed,
  Thermostat,
  Engineering,
  Warning
} from '@mui/icons-material';
import { bracketSampleProject } from '../../data/sampleMeshData';
import type { BoundaryCondition as BracketBoundaryCondition } from '../../data/bracketProjectData';

interface BoundaryCondition {
  id: string;
  name: string;
  type: 'displacement' | 'force' | 'pressure' | 'temperature' | 'heat_flux' | 'convection';
  location: 'face' | 'edge' | 'vertex' | 'volume';
  geometrySelection: string[];
  values: {
    x?: number;
    y?: number;
    z?: number;
    magnitude?: number;
    direction?: [number, number, number];
    temperature?: number;
    coefficient?: number;
  };
  coordinate_system: 'global' | 'local' | 'cylindrical';
  active: boolean;
}

interface BoundaryConditionsProps {
  open: boolean;
  onClose: () => void;
  onApply?: (conditions: BoundaryCondition[]) => void;
}

const convertBracketBoundaryConditions = (bracketConditions: BracketBoundaryCondition[]): BoundaryCondition[] => {
  return bracketConditions.map(condition => ({
    id: condition.id,
    name: condition.name,
    type: condition.type === 'load' ? 'force' : condition.type === 'fixed' || condition.type === 'pinned' ? 'displacement' : 'pressure',
    location: 'face' as const,
    geometrySelection: condition.surfaceIds || [],
    values: {
      magnitude: condition.magnitude,
      direction: [condition.direction.x, condition.direction.y, condition.direction.z] as [number, number, number],
      x: condition.direction.x * condition.magnitude,
      y: condition.direction.y * condition.magnitude,
      z: condition.direction.z * condition.magnitude
    },
    coordinate_system: 'global' as const,
    active: condition.isActive
  }));
};

const BoundaryConditions: React.FC<BoundaryConditionsProps> = ({ open, onClose, onApply }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedProject, setSelectedProject] = useState<'simple' | 'bracket'>('bracket');
  const [conditions, setConditions] = useState<BoundaryCondition[]>(convertBracketBoundaryConditions(bracketSampleProject.boundaryConditions));
  const [selectedCondition, setSelectedCondition] = useState<BoundaryCondition | null>(null);
  const [editMode, setEditMode] = useState<'add' | 'edit' | null>(null);

  const [newCondition, setNewCondition] = useState<Partial<BoundaryCondition>>({
    name: '',
    type: 'displacement',
    location: 'face',
    geometrySelection: [],
    values: {},
    coordinate_system: 'global',
    active: true
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleProjectChange = (project: 'simple' | 'bracket') => {
    setSelectedProject(project);
    if (project === 'bracket') {
      setConditions(convertBracketBoundaryConditions(bracketSampleProject.boundaryConditions));
    } else {
      setConditions([]);
    }
    setSelectedCondition(null);
    setEditMode(null);
  };

  const handleAddCondition = () => {
    setEditMode('add');
    setNewCondition({
      name: `BC_${conditions.length + 1}`,
      type: 'displacement',
      location: 'face',
      geometrySelection: [],
      values: {},
      coordinate_system: 'global',
      active: true
    });
  };

  const handleSaveCondition = () => {
    if (newCondition.name && newCondition.type) {
      const condition: BoundaryCondition = {
        id: editMode === 'edit' ? selectedCondition?.id || '' : `bc_${Date.now()}`,
        name: newCondition.name,
        type: newCondition.type,
        location: newCondition.location || 'face',
        geometrySelection: newCondition.geometrySelection || [],
        values: newCondition.values || {},
        coordinate_system: newCondition.coordinate_system || 'global',
        active: newCondition.active ?? true
      };

      if (editMode === 'add') {
        setConditions([...conditions, condition]);
      } else if (editMode === 'edit' && selectedCondition) {
        setConditions(conditions.map(c => c.id === selectedCondition.id ? condition : c));
      }

      setEditMode(null);
      setSelectedCondition(null);
    }
  };

  const handleEditCondition = (condition: BoundaryCondition) => {
    setSelectedCondition(condition);
    setNewCondition(condition);
    setEditMode('edit');
  };

  const handleDeleteCondition = (id: string) => {
    setConditions(conditions.filter(c => c.id !== id));
    if (selectedCondition?.id === id) {
      setSelectedCondition(null);
    }
  };

  const handleToggleActive = (id: string) => {
    setConditions(conditions.map(c =>
      c.id === id ? { ...c, active: !c.active } : c
    ));
  };

  const getConditionIcon = (type: string) => {
    switch (type) {
      case 'displacement': return <LocationOn />;
      case 'force': return <Speed />;
      case 'pressure': return <Engineering />;
      case 'temperature': return <Thermostat />;
      default: return <Security />;
    }
  };

  const getConditionColor = (type: string) => {
    switch (type) {
      case 'displacement': return 'primary';
      case 'force': return 'secondary';
      case 'pressure': return 'info';
      case 'temperature': return 'warning';
      default: return 'default';
    }
  };

  const getConditionDescription = (type: string) => {
    switch (type) {
      case 'displacement': return 'Fixed displacement or prescribed motion';
      case 'force': return 'Applied force or concentrated load';
      case 'pressure': return 'Distributed pressure load';
      case 'temperature': return 'Fixed temperature boundary';
      case 'heat_flux': return 'Applied heat flux';
      case 'convection': return 'Convective heat transfer';
      default: return '';
    }
  };

  const renderValueInputs = () => {
    if (!newCondition.type) return null;

    switch (newCondition.type) {
      case 'displacement':
        return (
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <TextField
                fullWidth
                label="X Displacement"
                type="number"
                value={newCondition.values?.x || 0}
                onChange={(e) => setNewCondition({
                  ...newCondition,
                  values: { ...newCondition.values, x: Number(e.target.value) }
                })}
                InputProps={{
                  endAdornment: <InputAdornment position="end">mm</InputAdornment>
                }}
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                fullWidth
                label="Y Displacement"
                type="number"
                value={newCondition.values?.y || 0}
                onChange={(e) => setNewCondition({
                  ...newCondition,
                  values: { ...newCondition.values, y: Number(e.target.value) }
                })}
                InputProps={{
                  endAdornment: <InputAdornment position="end">mm</InputAdornment>
                }}
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                fullWidth
                label="Z Displacement"
                type="number"
                value={newCondition.values?.z || 0}
                onChange={(e) => setNewCondition({
                  ...newCondition,
                  values: { ...newCondition.values, z: Number(e.target.value) }
                })}
                InputProps={{
                  endAdornment: <InputAdornment position="end">mm</InputAdornment>
                }}
              />
            </Grid>
          </Grid>
        );

      case 'force':
        return (
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Magnitude"
                type="number"
                value={newCondition.values?.magnitude || 0}
                onChange={(e) => setNewCondition({
                  ...newCondition,
                  values: { ...newCondition.values, magnitude: Number(e.target.value) }
                })}
                InputProps={{
                  endAdornment: <InputAdornment position="end">N</InputAdornment>
                }}
              />
            </Grid>
            <Grid item xs={2}>
              <TextField
                fullWidth
                label="Dir X"
                type="number"
                value={newCondition.values?.direction?.[0] || 0}
                onChange={(e) => {
                  const direction = newCondition.values?.direction || [0, 0, 0];
                  direction[0] = Number(e.target.value);
                  setNewCondition({
                    ...newCondition,
                    values: { ...newCondition.values, direction: direction as [number, number, number] }
                  });
                }}
              />
            </Grid>
            <Grid item xs={2}>
              <TextField
                fullWidth
                label="Dir Y"
                type="number"
                value={newCondition.values?.direction?.[1] || 0}
                onChange={(e) => {
                  const direction = newCondition.values?.direction || [0, 0, 0];
                  direction[1] = Number(e.target.value);
                  setNewCondition({
                    ...newCondition,
                    values: { ...newCondition.values, direction: direction as [number, number, number] }
                  });
                }}
              />
            </Grid>
            <Grid item xs={2}>
              <TextField
                fullWidth
                label="Dir Z"
                type="number"
                value={newCondition.values?.direction?.[2] || 1}
                onChange={(e) => {
                  const direction = newCondition.values?.direction || [0, 0, 0];
                  direction[2] = Number(e.target.value);
                  setNewCondition({
                    ...newCondition,
                    values: { ...newCondition.values, direction: direction as [number, number, number] }
                  });
                }}
              />
            </Grid>
          </Grid>
        );

      case 'pressure':
        return (
          <TextField
            fullWidth
            label="Pressure"
            type="number"
            value={newCondition.values?.magnitude || 0}
            onChange={(e) => setNewCondition({
              ...newCondition,
              values: { ...newCondition.values, magnitude: Number(e.target.value) }
            })}
            InputProps={{
              endAdornment: <InputAdornment position="end">MPa</InputAdornment>
            }}
          />
        );

      case 'temperature':
        return (
          <TextField
            fullWidth
            label="Temperature"
            type="number"
            value={newCondition.values?.temperature || 20}
            onChange={(e) => setNewCondition({
              ...newCondition,
              values: { ...newCondition.values, temperature: Number(e.target.value) }
            })}
            InputProps={{
              endAdornment: <InputAdornment position="end">°C</InputAdornment>
            }}
          />
        );

      case 'convection':
        return (
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Heat Transfer Coefficient"
                type="number"
                value={newCondition.values?.coefficient || 25}
                onChange={(e) => setNewCondition({
                  ...newCondition,
                  values: { ...newCondition.values, coefficient: Number(e.target.value) }
                })}
                InputProps={{
                  endAdornment: <InputAdornment position="end">W/m²K</InputAdornment>
                }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Ambient Temperature"
                type="number"
                value={newCondition.values?.temperature || 20}
                onChange={(e) => setNewCondition({
                  ...newCondition,
                  values: { ...newCondition.values, temperature: Number(e.target.value) }
                })}
                InputProps={{
                  endAdornment: <InputAdornment position="end">°C</InputAdornment>
                }}
              />
            </Grid>
          </Grid>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Security />
          Boundary Conditions
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ width: '100%' }}>
          <Box sx={{ mb: 3 }}>
            <FormControl fullWidth>
              <InputLabel>Sample Project</InputLabel>
              <Select
                value={selectedProject}
                label="Sample Project"
                onChange={(e) => handleProjectChange(e.target.value as 'simple' | 'bracket')}
              >
                <MenuItem value="simple">Simple Beam</MenuItem>
                <MenuItem value="bracket">L-Bracket (CAE)</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Manage Conditions" />
            <Tab label="Add/Edit Condition" />
            <Tab label="Summary" />
          </Tabs>

          {activeTab === 0 && (
            <Box sx={{ mt: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Applied Boundary Conditions</Typography>
                <Button variant="contained" startIcon={<Add />} onClick={handleAddCondition}>
                  Add Condition
                </Button>
              </Box>

              {conditions.length === 0 ? (
                <Alert severity="info">
                  No boundary conditions defined. Click "Add Condition" to get started.
                </Alert>
              ) : (
                <List>
                  {conditions.map((condition) => (
                    <Card key={condition.id} sx={{ mb: 2 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexGrow: 1 }}>
                            {getConditionIcon(condition.type)}
                            <Box>
                              <Typography variant="subtitle1" fontWeight="medium">
                                {condition.name}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {getConditionDescription(condition.type)}
                              </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', gap: 1 }}>
                              <Chip
                                label={condition.type}
                                color={getConditionColor(condition.type) as any}
                                size="small"
                              />
                              <Chip
                                label={condition.location}
                                variant="outlined"
                                size="small"
                              />
                              {!condition.active && (
                                <Chip
                                  label="Inactive"
                                  color="default"
                                  size="small"
                                />
                              )}
                            </Box>
                          </Box>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <FormControlLabel
                              control={
                                <Switch
                                  checked={condition.active}
                                  onChange={() => handleToggleActive(condition.id)}
                                />
                              }
                              label=""
                            />
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => handleEditCondition(condition)}
                            >
                              <Edit />
                            </IconButton>
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleDeleteCondition(condition.id)}
                            >
                              <Delete />
                            </IconButton>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  ))}
                </List>
              )}
            </Box>
          )}

          {activeTab === 1 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                {editMode === 'add' ? 'Add New' : 'Edit'} Boundary Condition
              </Typography>

              {editMode ? (
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Condition Name"
                      value={newCondition.name}
                      onChange={(e) => setNewCondition({ ...newCondition, name: e.target.value })}
                      margin="normal"
                    />
                    <FormControl fullWidth margin="normal">
                      <InputLabel>Condition Type</InputLabel>
                      <Select
                        value={newCondition.type}
                        label="Condition Type"
                        onChange={(e) => setNewCondition({ ...newCondition, type: e.target.value as any })}
                      >
                        <MenuItem value="displacement">Fixed Displacement</MenuItem>
                        <MenuItem value="force">Applied Force</MenuItem>
                        <MenuItem value="pressure">Pressure Load</MenuItem>
                        <MenuItem value="temperature">Fixed Temperature</MenuItem>
                        <MenuItem value="heat_flux">Heat Flux</MenuItem>
                        <MenuItem value="convection">Convection</MenuItem>
                      </Select>
                    </FormControl>
                    <FormControl fullWidth margin="normal">
                      <InputLabel>Apply to</InputLabel>
                      <Select
                        value={newCondition.location}
                        label="Apply to"
                        onChange={(e) => setNewCondition({ ...newCondition, location: e.target.value as any })}
                      >
                        <MenuItem value="face">Face/Surface</MenuItem>
                        <MenuItem value="edge">Edge/Line</MenuItem>
                        <MenuItem value="vertex">Vertex/Point</MenuItem>
                        <MenuItem value="volume">Volume/Body</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
                      Value Definition
                    </Typography>
                    {renderValueInputs()}
                    <FormControl fullWidth margin="normal">
                      <InputLabel>Coordinate System</InputLabel>
                      <Select
                        value={newCondition.coordinate_system}
                        label="Coordinate System"
                        onChange={(e) => setNewCondition({ ...newCondition, coordinate_system: e.target.value as any })}
                      >
                        <MenuItem value="global">Global Cartesian</MenuItem>
                        <MenuItem value="local">Local Coordinate System</MenuItem>
                        <MenuItem value="cylindrical">Cylindrical</MenuItem>
                      </Select>
                    </FormControl>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={newCondition.active}
                          onChange={(e) => setNewCondition({ ...newCondition, active: e.target.checked })}
                        />
                      }
                      label="Active"
                      sx={{ mt: 2 }}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Alert severity="info" sx={{ mb: 2 }}>
                      {getConditionDescription(newCondition.type || 'displacement')}
                    </Alert>
                    <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                      <Button onClick={() => setEditMode(null)}>Cancel</Button>
                      <Button variant="contained" onClick={handleSaveCondition}>
                        {editMode === 'add' ? 'Add' : 'Update'} Condition
                      </Button>
                    </Box>
                  </Grid>
                </Grid>
              ) : (
                <Alert severity="info">
                  Select a boundary condition to edit or click "Add Condition" to create a new one.
                </Alert>
              )}
            </Box>
          )}

          {activeTab === 2 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>Boundary Conditions Summary</Typography>

              {conditions.length === 0 ? (
                <Alert severity="warning">
                  <Warning sx={{ mr: 1 }} />
                  No boundary conditions defined. Your model may be under-constrained.
                </Alert>
              ) : (
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>Active Conditions</Typography>
                        <List dense>
                          {conditions.filter(c => c.active).map((condition) => (
                            <ListItem key={condition.id}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {getConditionIcon(condition.type)}
                                <ListItemText
                                  primary={condition.name}
                                  secondary={`${condition.type} on ${condition.location}`}
                                />
                              </Box>
                            </ListItem>
                          ))}
                        </List>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>Statistics</Typography>
                        <Box sx={{ '& > div': { display: 'flex', justifyContent: 'space-between', py: 0.5 } }}>
                          <div><span>Total Conditions:</span> <strong>{conditions.length}</strong></div>
                          <div><span>Active:</span> <strong>{conditions.filter(c => c.active).length}</strong></div>
                          <div><span>Displacement BCs:</span> <strong>{conditions.filter(c => c.type === 'displacement').length}</strong></div>
                          <div><span>Load BCs:</span> <strong>{conditions.filter(c => ['force', 'pressure'].includes(c.type)).length}</strong></div>
                          <div><span>Thermal BCs:</span> <strong>{conditions.filter(c => ['temperature', 'heat_flux', 'convection'].includes(c.type)).length}</strong></div>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              )}
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          onClick={() => {
            if (onApply) onApply(conditions);
            onClose();
          }}
        >
          Apply Conditions
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default BoundaryConditions;