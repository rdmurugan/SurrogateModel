import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Tabs,
  Tab,
  Card,
  CardContent,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  Science,
  Add,
  Edit,
  Delete,
  Assignment,
  Info,
  Warning,
  CheckCircle,
  AutoAwesome,
  Download,
  Upload
} from '@mui/icons-material';
import { sampleMaterials, sampleBodies, MaterialData, GeometryBody, bracketSampleProject } from '../../data/sampleMeshData';
import type { BracketMaterial, BracketGeometryPart } from '../../data/bracketProjectData';

interface MaterialsAssignmentProps {
  open: boolean;
  onClose: () => void;
  onAssignMaterial?: (bodyId: string, materialId: string) => void;
}

const convertBracketMaterialsToMaterialData = (bracketMaterials: BracketMaterial[]): MaterialData[] => {
  return bracketMaterials.map(material => ({
    id: material.id,
    name: material.name,
    type: material.type as 'metal' | 'polymer' | 'composite' | 'ceramic',
    properties: {
      density: material.properties.density,
      youngsModulus: material.properties.youngsModulus,
      poissonsRatio: material.properties.poissonsRatio,
      yieldStrength: material.properties.yieldStrength,
      ultimateStrength: material.properties.ultimateStrength,
      thermalConductivity: material.properties.thermalConductivity,
      thermalExpansion: material.properties.thermalExpansion,
      specificHeat: material.properties.specificHeat
    },
    color: material.color,
    isAssigned: false,
    assignedBodies: [],
    elementIds: [],
    volume: 0,
    status: 'valid' as const,
    description: `${material.supplier} - ${material.name}`
  }));
};

const convertBracketPartsToGeometryBodies = (bracketParts: BracketGeometryPart[]): GeometryBody[] => {
  return bracketParts.map(part => ({
    id: part.id,
    name: part.name,
    type: 'solid' as const,
    volume: part.volume,
    surfaceArea: part.surfaceArea,
    materialId: part.materialId,
    elementIds: [],
    visible: true,
    color: part.color
  }));
};

const MaterialsAssignment: React.FC<MaterialsAssignmentProps> = ({
  open,
  onClose,
  onAssignMaterial
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedProject, setSelectedProject] = useState<'simple' | 'bracket'>('bracket');
  const [materials, setMaterials] = useState<MaterialData[]>(convertBracketMaterialsToMaterialData(bracketSampleProject.materials));
  const [bodies, setBodies] = useState<GeometryBody[]>(convertBracketPartsToGeometryBodies(bracketSampleProject.geometry.parts));
  const [selectedMaterial, setSelectedMaterial] = useState<string | null>(null);
  const [selectedBody, setSelectedBody] = useState<string | null>(null);
  const [autoAssignProgress, setAutoAssignProgress] = useState(0);
  const [isAutoAssigning, setIsAutoAssigning] = useState(false);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleProjectChange = (project: 'simple' | 'bracket') => {
    setSelectedProject(project);
    if (project === 'bracket') {
      setMaterials(convertBracketMaterialsToMaterialData(bracketSampleProject.materials));
      setBodies(convertBracketPartsToGeometryBodies(bracketSampleProject.geometry.parts));
    } else {
      setMaterials(sampleMaterials);
      setBodies(sampleBodies);
    }
    setSelectedMaterial(null);
    setSelectedBody(null);
  };

  const handleAssignMaterial = (bodyId: string, materialId: string) => {
    setBodies(prev => prev.map(body =>
      body.id === bodyId
        ? { ...body, materialId, color: materials.find(m => m.id === materialId)?.color || body.color }
        : body
    ));

    setMaterials(prev => prev.map(material => ({
      ...material,
      isAssigned: material.id === materialId ? true : material.isAssigned,
      assignedBodies: material.id === materialId
        ? [...material.assignedBodies.filter(name => name !== bodies.find(b => b.id === bodyId)?.name), bodies.find(b => b.id === bodyId)?.name || '']
        : material.assignedBodies.filter(name => name !== bodies.find(b => b.id === bodyId)?.name)
    })));

    if (onAssignMaterial) {
      onAssignMaterial(bodyId, materialId);
    }
  };

  const handleRemoveMaterial = (bodyId: string) => {
    const body = bodies.find(b => b.id === bodyId);
    if (body?.materialId) {
      setMaterials(prev => prev.map(material =>
        material.id === body.materialId
          ? { ...material, assignedBodies: material.assignedBodies.filter(name => name !== body.name) }
          : material
      ));
    }

    setBodies(prev => prev.map(body =>
      body.id === bodyId
        ? { ...body, materialId: undefined, color: '#f44336' }
        : body
    ));
  };

  const handleAutoAssign = () => {
    setIsAutoAssigning(true);
    setAutoAssignProgress(0);

    const interval = setInterval(() => {
      setAutoAssignProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAutoAssigning(false);

          // Auto-assign materials based on volume (larger volumes get steel, smaller get aluminum)
          setBodies(prev => prev.map(body => {
            if (!body.materialId) {
              const material = body.volume > 10 ? 'steel' : 'aluminum';
              return { ...body, materialId: material, color: materials.find(m => m.id === material)?.color || body.color };
            }
            return body;
          }));

          return 100;
        }
        return prev + Math.random() * 20 + 10;
      });
    }, 300);
  };

  const getAssignmentStatus = () => {
    const unassigned = bodies.filter(body => !body.materialId).length;
    const total = bodies.length;
    return { assigned: total - unassigned, total, percentage: ((total - unassigned) / total) * 100 };
  };

  const renderAssignmentTab = () => {
    const status = getAssignmentStatus();

    return (
      <Box sx={{ mt: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Available Materials</Typography>
                  <Button startIcon={<Add />} size="small" variant="outlined">
                    Add Material
                  </Button>
                </Box>

                <List dense>
                  {materials.map((material) => (
                    <ListItem
                      key={material.id}
                      sx={{
                        border: '1px solid #e0e0e0',
                        borderRadius: 1,
                        mb: 1,
                        backgroundColor: selectedMaterial === material.id ? 'rgba(25, 118, 210, 0.1)' : 'transparent',
                        cursor: 'pointer'
                      }}
                      onClick={() => setSelectedMaterial(material.id)}
                    >
                      <ListItemIcon>
                        <Box
                          sx={{
                            width: 20,
                            height: 20,
                            backgroundColor: material.color,
                            borderRadius: '50%',
                            border: '1px solid #ccc'
                          }}
                        />
                      </ListItemIcon>
                      <ListItemText
                        primary={material.name}
                        secondary={`${material.type} • ${material.assignedBodies.length} bodies assigned`}
                      />
                      <ListItemSecondaryAction>
                        <Chip
                          size="small"
                          label={material.status}
                          color={material.status === 'valid' ? 'success' : material.status === 'warning' ? 'warning' : 'error'}
                        />
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Geometry Bodies</Typography>
                  <Button
                    startIcon={<AutoAwesome />}
                    size="small"
                    variant="outlined"
                    onClick={handleAutoAssign}
                  >
                    Auto Assign
                  </Button>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Assignment Progress: {status.assigned}/{status.total} bodies
                  </Typography>
                  <LinearProgress variant="determinate" value={status.percentage} />
                </Box>

                <List dense>
                  {bodies.map((body) => (
                    <ListItem
                      key={body.id}
                      sx={{
                        border: '1px solid #e0e0e0',
                        borderRadius: 1,
                        mb: 1,
                        backgroundColor: selectedBody === body.id ? 'rgba(25, 118, 210, 0.1)' : 'transparent',
                        cursor: 'pointer'
                      }}
                      onClick={() => setSelectedBody(body.id)}
                    >
                      <ListItemIcon>
                        <Box
                          sx={{
                            width: 20,
                            height: 20,
                            backgroundColor: body.color,
                            borderRadius: '50%',
                            border: '1px solid #ccc'
                          }}
                        />
                      </ListItemIcon>
                      <ListItemText
                        primary={body.name}
                        secondary={`Volume: ${body.volume.toFixed(2)} mm³ • ${
                          body.materialId
                            ? materials.find(m => m.id === body.materialId)?.name || 'Unknown'
                            : 'No material assigned'
                        }`}
                      />
                      <ListItemSecondaryAction>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          {selectedMaterial && body.id === selectedBody && (
                            <Button
                              size="small"
                              variant="contained"
                              onClick={() => handleAssignMaterial(body.id, selectedMaterial)}
                            >
                              Assign
                            </Button>
                          )}
                          {body.materialId && (
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleRemoveMaterial(body.id)}
                            >
                              <Delete />
                            </IconButton>
                          )}
                        </Box>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {status.assigned < status.total && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            <Typography variant="body2">
              {status.total - status.assigned} geometry bodies do not have materials assigned.
              This may cause analysis errors.
            </Typography>
          </Alert>
        )}
      </Box>
    );
  };

  const renderPropertiesTab = () => (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>Material Properties</Typography>

      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Material</TableCell>
              <TableCell align="right">Density (kg/m³)</TableCell>
              <TableCell align="right">Young's Modulus (MPa)</TableCell>
              <TableCell align="right">Poisson's Ratio</TableCell>
              <TableCell align="right">Yield Strength (MPa)</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {materials.map((material) => (
              <TableRow key={material.id}>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box
                      sx={{
                        width: 16,
                        height: 16,
                        backgroundColor: material.color,
                        borderRadius: '50%',
                        border: '1px solid #ccc'
                      }}
                    />
                    {material.name}
                  </Box>
                </TableCell>
                <TableCell align="right">{material.properties.density.toLocaleString()}</TableCell>
                <TableCell align="right">{material.properties.youngsModulus.toLocaleString()}</TableCell>
                <TableCell align="right">{material.properties.poissonsRatio}</TableCell>
                <TableCell align="right">
                  {material.properties.yieldStrength?.toLocaleString() || 'N/A'}
                </TableCell>
                <TableCell align="center">
                  <IconButton size="small">
                    <Edit />
                  </IconButton>
                  <IconButton size="small" color="error">
                    <Delete />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {selectedMaterial && (
        <Card sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {materials.find(m => m.id === selectedMaterial)?.name} - Detailed Properties
            </Typography>
            <Grid container spacing={2}>
              {Object.entries(materials.find(m => m.id === selectedMaterial)?.properties || {}).map(([key, value]) => (
                <Grid item xs={6} md={4} key={key}>
                  <Typography variant="body2" color="text.secondary">
                    {key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                  </Typography>
                  <Typography variant="body1" fontWeight={600}>
                    {typeof value === 'number' ? value.toLocaleString() : value}
                  </Typography>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );

  const renderValidationTab = () => {
    const unassignedBodies = bodies.filter(body => !body.materialId);
    const invalidMaterials = materials.filter(material => material.status !== 'valid');

    return (
      <Box sx={{ mt: 3 }}>
        <Typography variant="h6" gutterBottom>Material Assignment Validation</Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CheckCircle color="success" />
                  Valid Assignments
                </Typography>
                <List dense>
                  {bodies.filter(body => body.materialId).map(body => (
                    <ListItem key={body.id}>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary={body.name}
                        secondary={materials.find(m => m.id === body.materialId)?.name}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Warning color="warning" />
                  Issues Found
                </Typography>

                {unassignedBodies.length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" fontWeight={600} gutterBottom>
                      Unassigned Bodies:
                    </Typography>
                    {unassignedBodies.map(body => (
                      <Chip
                        key={body.id}
                        label={body.name}
                        color="error"
                        size="small"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </Box>
                )}

                {invalidMaterials.length > 0 && (
                  <Box>
                    <Typography variant="body2" fontWeight={600} gutterBottom>
                      Invalid Materials:
                    </Typography>
                    {invalidMaterials.map(material => (
                      <Chip
                        key={material.id}
                        label={material.name}
                        color={material.status === 'warning' ? 'warning' : 'error'}
                        size="small"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </Box>
                )}

                {unassignedBodies.length === 0 && invalidMaterials.length === 0 && (
                  <Typography variant="body2" color="text.secondary">
                    No issues found. All materials are properly assigned.
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
          <Button startIcon={<Download />} variant="outlined">
            Export Assignment Report
          </Button>
          <Button startIcon={<Upload />} variant="outlined">
            Import Material Library
          </Button>
        </Box>
      </Box>
    );
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Science />
          Materials Assignment
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
            <Tab label="Assignment" icon={<Assignment />} />
            <Tab label="Properties" icon={<Info />} />
            <Tab label="Validation" icon={<CheckCircle />} />
          </Tabs>

          {activeTab === 0 && renderAssignmentTab()}
          {activeTab === 1 && renderPropertiesTab()}
          {activeTab === 2 && renderValidationTab()}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        <Button variant="contained">Apply Changes</Button>
      </DialogActions>
    </Dialog>
  );
};

export default MaterialsAssignment;