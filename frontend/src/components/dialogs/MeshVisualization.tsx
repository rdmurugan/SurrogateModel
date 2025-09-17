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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  Grid,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Divider,
  Alert,
  TextField
} from '@mui/material';
import {
  GridOn,
  Palette,
  Speed,
  Assessment,
  ZoomIn,
  ZoomOut,
  CenterFocusStrong,
  Refresh,
  Download,
  Settings
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { sampleMeshData, MeshData, bracketSampleProject } from '../../data/sampleMeshData';

interface MeshVisualizationProps {
  open: boolean;
  onClose: () => void;
  meshData?: MeshData;
}

interface MeshSettings {
  displayMode: 'solid' | 'wireframe' | 'points' | 'hybrid';
  showEdges: boolean;
  showNodes: boolean;
  transparency: number;
  colorMap: 'quality' | 'size' | 'aspect' | 'skewness' | 'material';
  qualityThreshold: number;
  edgeThickness: number;
  nodeSize: number;
  lighting: boolean;
  shadows: boolean;
  autoRotate: boolean;
  backgroundColor: string;
}

const MeshVisualization: React.FC<MeshVisualizationProps> = ({
  open,
  onClose,
  meshData = sampleMeshData
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedProject, setSelectedProject] = useState<'simple' | 'bracket'>('bracket');
  const [meshSettings, setMeshSettings] = useState<MeshSettings>({
    displayMode: 'solid',
    showEdges: true,
    showNodes: false,
    transparency: 0,
    colorMap: 'quality',
    qualityThreshold: 0.3,
    edgeThickness: 1,
    nodeSize: 2,
    lighting: true,
    shadows: false,
    autoRotate: false,
    backgroundColor: '#2d2d30'
  });

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);

  // Get current mesh data based on selected project
  const currentMeshData = selectedProject === 'bracket' ? bracketSampleProject.mesh : meshData;

  // Use real mesh stats from current data
  const meshStats = {
    totalElements: currentMeshData.elementCount,
    totalNodes: currentMeshData.nodeCount,
    avgQuality: currentMeshData.qualityStats.average,
    minQuality: currentMeshData.qualityStats.min,
    maxQuality: currentMeshData.qualityStats.max,
    elementTypes: currentMeshData.elementTypes,
    qualityDistribution: currentMeshData.qualityStats.distribution
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleSettingChange = (setting: keyof MeshSettings, value: any) => {
    setMeshSettings(prev => ({ ...prev, [setting]: value }));
  };

  const handleAnalyzeMesh = () => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);

    const interval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAnalyzing(false);
          return 100;
        }
        return prev + Math.random() * 15 + 5;
      });
    }, 200);
  };

  const handleExportMesh = () => {
    console.log('Exporting mesh data...');
    // TODO: Implement mesh export functionality
  };

  const handleResetView = () => {
    console.log('Resetting view...');
    // TODO: Implement view reset
  };

  const getQualityColor = (quality: number) => {
    if (quality >= 0.8) return 'success';
    if (quality >= 0.6) return 'warning';
    return 'error';
  };

  // Generate 3D mesh visualization data for Plotly
  const generate3DMeshData = () => {
    const { nodes, elements } = currentMeshData;

    // Create node scatter plot
    const nodeTrace = {
      x: nodes.map(n => n.x),
      y: nodes.map(n => n.y),
      z: nodes.map(n => n.z),
      mode: 'markers',
      type: 'scatter3d',
      marker: {
        size: meshSettings.nodeSize,
        color: meshSettings.showNodes ? '#00ff00' : 'transparent',
        opacity: meshSettings.showNodes ? 0.8 : 0
      },
      name: 'Nodes',
      visible: meshSettings.showNodes
    };

    // Create element edges for wireframe/solid display
    const edgeTraces: any[] = [];
    if (meshSettings.showEdges || meshSettings.displayMode === 'wireframe') {
      elements.forEach((element, idx) => {
        // For tetrahedra, draw edges
        if (element.type === 'tetrahedral' && element.nodeIds.length === 4) {
          const elementNodes = element.nodeIds.map(id => nodes.find(n => n.id === id)).filter(Boolean) as any[];
          if (elementNodes.length === 4) {
            // Tetrahedral edges: connect all pairs
            const edges = [
              [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
            ];

            edges.forEach(([i, j]) => {
              if (elementNodes[i] && elementNodes[j]) {
                edgeTraces.push({
                  x: [elementNodes[i].x, elementNodes[j].x, null],
                  y: [elementNodes[i].y, elementNodes[j].y, null],
                  z: [elementNodes[i].z, elementNodes[j].z, null],
                  mode: 'lines',
                  type: 'scatter3d',
                  line: {
                    width: meshSettings.edgeThickness,
                    color: getElementColor(element)
                  },
                  showlegend: false,
                  hoverinfo: 'skip'
                });
              }
            });
          }
        }
      });
    }

    // Create surface mesh for solid display
    const surfaceTrace = meshSettings.displayMode === 'solid' || meshSettings.displayMode === 'hybrid' ?
      generateSurfaceMesh() : null;

    return {
      data: [
        nodeTrace,
        ...edgeTraces.slice(0, 100), // Limit edges for performance
        ...(surfaceTrace ? [surfaceTrace] : [])
      ].filter(Boolean),
      layout: {
        scene: {
          xaxis: { title: 'X (mm)', backgroundcolor: meshSettings.backgroundColor },
          yaxis: { title: 'Y (mm)', backgroundcolor: meshSettings.backgroundColor },
          zaxis: { title: 'Z (mm)', backgroundcolor: meshSettings.backgroundColor },
          bgcolor: meshSettings.backgroundColor,
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 }
          }
        },
        paper_bgcolor: meshSettings.backgroundColor,
        plot_bgcolor: meshSettings.backgroundColor,
        font: { color: '#ffffff' },
        margin: { l: 0, r: 0, t: 30, b: 0 },
        showlegend: false
      },
      config: {
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d'] as any,
        displaylogo: false
      }
    };
  };

  const getElementColor = (element: any) => {
    switch (meshSettings.colorMap) {
      case 'quality':
        if (element.quality >= 0.8) return '#00ff00';
        if (element.quality >= 0.6) return '#ffff00';
        if (element.quality >= 0.4) return '#ff8800';
        return '#ff0000';
      case 'material':
        return element.materialId === 'steel' ? '#4a5568' : '#a0aec0';
      default:
        return '#888888';
    }
  };

  const generateSurfaceMesh = () => {
    // Generate a simplified surface representation
    const { nodes } = currentMeshData;
    const step = Math.max(1, Math.floor(nodes.length / 1000)); // Subsample for performance

    const sampledNodes = nodes.filter((_, idx) => idx % step === 0);

    return {
      x: sampledNodes.map(n => n.x),
      y: sampledNodes.map(n => n.y),
      z: sampledNodes.map(n => n.z),
      type: 'mesh3d',
      opacity: 1 - (meshSettings.transparency / 100),
      color: '#4a90e2',
      flatshading: !meshSettings.lighting,
      lighting: {
        ambient: meshSettings.lighting ? 0.4 : 1,
        diffuse: meshSettings.lighting ? 0.6 : 0,
        specular: meshSettings.lighting ? 0.2 : 0
      },
      name: 'Mesh Surface',
      hovertemplate: 'Element Quality: %{z:.3f}<extra></extra>'
    };
  };

  const renderDisplayTab = () => (
    <Box sx={{ mt: 3 }}>
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Typography variant="h6" gutterBottom>3D Mesh Visualization</Typography>
          <Card sx={{ height: '500px', mb: 2 }}>
            <CardContent sx={{ height: '100%', p: 1 }}>
              <Plot
                data={generate3DMeshData().data}
                layout={generate3DMeshData().layout}
                config={generate3DMeshData().config}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Typography variant="h6" gutterBottom>Display Settings</Typography>

          <FormControl fullWidth margin="normal">
            <InputLabel>Sample Project</InputLabel>
            <Select
              value={selectedProject}
              label="Sample Project"
              onChange={(e) => setSelectedProject(e.target.value as 'simple' | 'bracket')}
            >
              <MenuItem value="simple">Simple Beam</MenuItem>
              <MenuItem value="bracket">L-Bracket (CAE)</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth margin="normal">
            <InputLabel>Display Mode</InputLabel>
            <Select
              value={meshSettings.displayMode}
              label="Display Mode"
              onChange={(e) => handleSettingChange('displayMode', e.target.value)}
            >
              <MenuItem value="solid">Solid</MenuItem>
              <MenuItem value="wireframe">Wireframe</MenuItem>
              <MenuItem value="points">Points</MenuItem>
              <MenuItem value="hybrid">Hybrid</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth margin="normal">
            <InputLabel>Color Mapping</InputLabel>
            <Select
              value={meshSettings.colorMap}
              label="Color Mapping"
              onChange={(e) => handleSettingChange('colorMap', e.target.value)}
            >
              <MenuItem value="quality">Element Quality</MenuItem>
              <MenuItem value="size">Element Size</MenuItem>
              <MenuItem value="aspect">Aspect Ratio</MenuItem>
              <MenuItem value="skewness">Skewness</MenuItem>
              <MenuItem value="material">Material</MenuItem>
            </Select>
          </FormControl>

          <Box sx={{ mt: 2 }}>
            <Typography gutterBottom>Transparency</Typography>
            <Slider
              value={meshSettings.transparency}
              min={0}
              max={100}
              valueLabelDisplay="auto"
              onChange={(e, value) => handleSettingChange('transparency', value)}
            />
          </Box>

          <Box sx={{ mt: 2 }}>
            <Typography gutterBottom>Quality Threshold</Typography>
            <Slider
              value={meshSettings.qualityThreshold}
              min={0}
              max={1}
              step={0.1}
              valueLabelDisplay="auto"
              onChange={(e, value) => handleSettingChange('qualityThreshold', value)}
            />
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Visibility Options</Typography>

          <FormControlLabel
            control={
              <Switch
                checked={meshSettings.showEdges}
                onChange={(e) => handleSettingChange('showEdges', e.target.checked)}
              />
            }
            label="Show Edges"
          />

          <FormControlLabel
            control={
              <Switch
                checked={meshSettings.showNodes}
                onChange={(e) => handleSettingChange('showNodes', e.target.checked)}
              />
            }
            label="Show Nodes"
          />

          <FormControlLabel
            control={
              <Switch
                checked={meshSettings.lighting}
                onChange={(e) => handleSettingChange('lighting', e.target.checked)}
              />
            }
            label="Lighting"
          />

          <FormControlLabel
            control={
              <Switch
                checked={meshSettings.shadows}
                onChange={(e) => handleSettingChange('shadows', e.target.checked)}
              />
            }
            label="Shadows"
          />

          <FormControlLabel
            control={
              <Switch
                checked={meshSettings.autoRotate}
                onChange={(e) => handleSettingChange('autoRotate', e.target.checked)}
              />
            }
            label="Auto Rotate"
          />

          {meshSettings.showEdges && (
            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>Edge Thickness</Typography>
              <Slider
                value={meshSettings.edgeThickness}
                min={0.5}
                max={5}
                step={0.5}
                valueLabelDisplay="auto"
                onChange={(e, value) => handleSettingChange('edgeThickness', value)}
              />
            </Box>
          )}

          {meshSettings.showNodes && (
            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>Node Size</Typography>
              <Slider
                value={meshSettings.nodeSize}
                min={1}
                max={10}
                valueLabelDisplay="auto"
                onChange={(e, value) => handleSettingChange('nodeSize', value)}
              />
            </Box>
          )}
        </Grid>
      </Grid>
    </Box>
  );

  const renderQualityTab = () => (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>Mesh Quality Analysis</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Quality Statistics</Typography>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography>Average Quality:</Typography>
                <Chip
                  label={meshStats.avgQuality.toFixed(3)}
                  color={getQualityColor(meshStats.avgQuality)}
                  size="small"
                />
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography>Minimum Quality:</Typography>
                <Chip
                  label={meshStats.minQuality.toFixed(3)}
                  color={getQualityColor(meshStats.minQuality)}
                  size="small"
                />
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography>Maximum Quality:</Typography>
                <Chip
                  label={meshStats.maxQuality.toFixed(3)}
                  color={getQualityColor(meshStats.maxQuality)}
                  size="small"
                />
              </Box>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle1" gutterBottom>Quality Distribution</Typography>
              {Object.entries(meshStats.qualityDistribution).map(([level, percentage]) => (
                <Box key={level} sx={{ mb: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                      {level}
                    </Typography>
                    <Typography variant="body2">{percentage}%</Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={percentage}
                    sx={{ height: 6, borderRadius: 3 }}
                  />
                </Box>
              ))}
            </CardContent>
          </Card>

          {isAnalyzing && (
            <Card sx={{ mt: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Analyzing Mesh Quality...</Typography>
                <LinearProgress variant="determinate" value={analysisProgress} />
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Progress: {Math.round(analysisProgress)}%
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Element Types</Typography>
              {Object.entries(meshStats.elementTypes).map(([type, count]) => (
                <Box key={type} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                    {type}:
                  </Typography>
                  <Typography variant="body2" fontWeight={600}>
                    {count.toLocaleString()}
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>

          <Box sx={{ mt: 2 }}>
            <Button
              fullWidth
              variant="contained"
              startIcon={<Assessment />}
              onClick={handleAnalyzeMesh}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Quality'}
            </Button>
          </Box>
        </Grid>
      </Grid>

      {meshStats.minQuality < 0.5 && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          <Typography variant="body2">
            Some elements have poor quality (below 0.5). Consider mesh refinement in these areas.
          </Typography>
        </Alert>
      )}
    </Box>
  );

  const renderControlsTab = () => (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>Viewport Controls</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>View Controls</Typography>

              <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <Button variant="outlined" startIcon={<ZoomIn />} size="small">
                  Zoom In
                </Button>
                <Button variant="outlined" startIcon={<ZoomOut />} size="small">
                  Zoom Out
                </Button>
                <Button variant="outlined" startIcon={<CenterFocusStrong />} size="small">
                  Fit All
                </Button>
              </Box>

              <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <Button variant="outlined" startIcon={<Refresh />} size="small" onClick={handleResetView}>
                  Reset View
                </Button>
                <Button variant="outlined" startIcon={<GridOn />} size="small">
                  Show Grid
                </Button>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>Background Color</Typography>
              <TextField
                type="color"
                value={meshSettings.backgroundColor}
                onChange={(e) => handleSettingChange('backgroundColor', e.target.value)}
                size="small"
                fullWidth
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>Export Options</Typography>

              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <Download />
                  </ListItemIcon>
                  <ListItemText
                    primary="Export Image"
                    secondary="PNG, JPG formats"
                  />
                  <ListItemSecondaryAction>
                    <Button size="small" variant="outlined">Export</Button>
                  </ListItemSecondaryAction>
                </ListItem>

                <ListItem>
                  <ListItemIcon>
                    <Download />
                  </ListItemIcon>
                  <ListItemText
                    primary="Export Mesh Data"
                    secondary="VTK, STL formats"
                  />
                  <ListItemSecondaryAction>
                    <Button size="small" variant="outlined" onClick={handleExportMesh}>Export</Button>
                  </ListItemSecondaryAction>
                </ListItem>

                <ListItem>
                  <ListItemIcon>
                    <Assessment />
                  </ListItemIcon>
                  <ListItemText
                    primary="Quality Report"
                    secondary="PDF report"
                  />
                  <ListItemSecondaryAction>
                    <Button size="small" variant="outlined">Generate</Button>
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Speed />
          Mesh Visualization & Analysis
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ width: '100%' }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Display" icon={<Palette />} />
            <Tab label="Quality" icon={<Assessment />} />
            <Tab label="Controls" icon={<Settings />} />
          </Tabs>

          {activeTab === 0 && renderDisplayTab()}
          {activeTab === 1 && renderQualityTab()}
          {activeTab === 2 && renderControlsTab()}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        <Button variant="contained">Apply Settings</Button>
      </DialogActions>
    </Dialog>
  );
};

export default MeshVisualization;