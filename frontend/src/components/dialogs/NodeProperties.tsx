import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Typography,
  Box,
  Tabs,
  Tab,
  Grid,
  Switch,
  FormControlLabel,
  Chip,
  Divider,
  Alert,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton
} from '@mui/material';
import {
  Edit,
  Visibility,
  Settings,
  Info,
  Warning,
  Speed,
  Engineering,
  Science,
  Delete,
  Add
} from '@mui/icons-material';

interface NodePropertiesProps {
  open: boolean;
  onClose: () => void;
  nodeId: string | null;
  nodeType: string;
  onUpdate?: (nodeId: string, properties: any) => void;
}

interface NodeProperty {
  key: string;
  value: any;
  type: 'string' | 'number' | 'boolean' | 'select' | 'units';
  options?: string[];
  unit?: string;
  description?: string;
}

const NodeProperties: React.FC<NodePropertiesProps> = ({
  open,
  onClose,
  nodeId,
  nodeType,
  onUpdate
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [properties, setProperties] = useState<Record<string, any>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Sample properties based on node type
  const getNodeProperties = (type: string): NodeProperty[] => {
    switch (type) {
      case 'project':
        return [
          { key: 'name', value: 'Structural Analysis Project', type: 'string', description: 'Project name' },
          { key: 'description', value: 'Advanced structural analysis of mechanical components', type: 'string', description: 'Project description' },
          { key: 'analysisType', value: 'Static Structural', type: 'select', options: ['Static Structural', 'Modal', 'Harmonic', 'Transient', 'Thermal'], description: 'Type of analysis' },
          { key: 'units', value: 'Metric (mm, kg, N, s, V, A)', type: 'select', options: ['Metric (mm, kg, N, s, V, A)', 'Metric (m, kg, N, s, V, A)', 'US Customary (in, lbm, lbf, s, V, A)'], description: 'Unit system' },
          { key: 'solver', value: 'Direct Sparse', type: 'select', options: ['Direct Sparse', 'Iterative', 'Distributed Sparse'], description: 'Solver type' },
          { key: 'autoSave', value: true, type: 'boolean', description: 'Enable auto-save' },
          { key: 'maxMemory', value: 4096, type: 'number', unit: 'MB', description: 'Maximum memory usage' }
        ];
      case 'geometry':
        return [
          { key: 'name', value: 'Main Assembly', type: 'string', description: 'Geometry name' },
          { key: 'material', value: 'Structural Steel', type: 'select', options: ['Structural Steel', 'Aluminum Alloy', 'Titanium', 'Carbon Steel'], description: 'Default material' },
          { key: 'volume', value: 125.67, type: 'number', unit: 'mm³', description: 'Total volume' },
          { key: 'surfaceArea', value: 892.3, type: 'number', unit: 'mm²', description: 'Total surface area' },
          { key: 'visible', value: true, type: 'boolean', description: 'Visibility in viewport' },
          { key: 'transparency', value: 0.0, type: 'number', unit: '%', description: 'Transparency level' },
          { key: 'color', value: 'Default', type: 'select', options: ['Default', 'Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan'], description: 'Display color' }
        ];
      case 'mesh':
        return [
          { key: 'name', value: 'Main Mesh', type: 'string', description: 'Mesh name' },
          { key: 'elementType', value: 'Tetrahedral', type: 'select', options: ['Tetrahedral', 'Hexahedral', 'Mixed'], description: 'Element type' },
          { key: 'elementSize', value: 5.0, type: 'number', unit: 'mm', description: 'Global element size' },
          { key: 'nodes', value: 67293, type: 'number', description: 'Number of nodes' },
          { key: 'elements', value: 125847, type: 'number', description: 'Number of elements' },
          { key: 'minQuality', value: 0.342, type: 'number', description: 'Minimum element quality' },
          { key: 'avgQuality', value: 0.876, type: 'number', description: 'Average element quality' },
          { key: 'refinement', value: true, type: 'boolean', description: 'Enable local refinement' }
        ];
      case 'file':
        return [
          { key: 'name', value: 'Applied Force', type: 'string', description: 'Load name' },
          { key: 'type', value: 'Force', type: 'select', options: ['Force', 'Pressure', 'Displacement', 'Temperature'], description: 'Load type' },
          { key: 'magnitude', value: 1000, type: 'number', unit: 'N', description: 'Load magnitude' },
          { key: 'direction', value: 'Y Negative', type: 'select', options: ['X Positive', 'X Negative', 'Y Positive', 'Y Negative', 'Z Positive', 'Z Negative'], description: 'Load direction' },
          { key: 'coordinateSystem', value: 'Global', type: 'select', options: ['Global', 'Local'], description: 'Coordinate system' },
          { key: 'active', value: true, type: 'boolean', description: 'Load active status' }
        ];
      default:
        return [
          { key: 'name', value: 'Item', type: 'string', description: 'Item name' },
          { key: 'visible', value: true, type: 'boolean', description: 'Visibility' }
        ];
    }
  };

  const [nodeProperties, setNodeProperties] = useState<NodeProperty[]>([]);

  useEffect(() => {
    if (nodeId && nodeType) {
      const props = getNodeProperties(nodeType);
      setNodeProperties(props);
      const initialProps: Record<string, any> = {};
      props.forEach(prop => {
        initialProps[prop.key] = prop.value;
      });
      setProperties(initialProps);
    }
  }, [nodeId, nodeType]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handlePropertyChange = (key: string, value: any) => {
    setProperties(prev => ({ ...prev, [key]: value }));
    // Clear error for this field
    if (errors[key]) {
      setErrors(prev => ({ ...prev, [key]: '' }));
    }
  };

  const validateProperties = () => {
    const newErrors: Record<string, string> = {};

    nodeProperties.forEach(prop => {
      const value = properties[prop.key];
      if (prop.type === 'string' && (!value || value.trim() === '')) {
        newErrors[prop.key] = 'This field is required';
      }
      if (prop.type === 'number' && (isNaN(value) || value < 0)) {
        newErrors[prop.key] = 'Must be a valid positive number';
      }
    });

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSave = () => {
    if (validateProperties()) {
      if (onUpdate && nodeId) {
        onUpdate(nodeId, properties);
      }
      onClose();
    }
  };

  const renderPropertyInput = (prop: NodeProperty) => {
    const value = properties[prop.key];
    const error = errors[prop.key];

    switch (prop.type) {
      case 'string':
        return (
          <TextField
            fullWidth
            label={prop.key.charAt(0).toUpperCase() + prop.key.slice(1)}
            value={value || ''}
            onChange={(e) => handlePropertyChange(prop.key, e.target.value)}
            error={!!error}
            helperText={error || prop.description}
            margin="normal"
          />
        );
      case 'number':
        return (
          <TextField
            fullWidth
            label={prop.key.charAt(0).toUpperCase() + prop.key.slice(1)}
            type="number"
            value={value || 0}
            onChange={(e) => handlePropertyChange(prop.key, parseFloat(e.target.value) || 0)}
            error={!!error}
            helperText={error || prop.description}
            margin="normal"
            InputProps={{
              endAdornment: prop.unit && <Typography variant="caption">{prop.unit}</Typography>
            }}
          />
        );
      case 'boolean':
        return (
          <FormControlLabel
            control={
              <Switch
                checked={value || false}
                onChange={(e) => handlePropertyChange(prop.key, e.target.checked)}
              />
            }
            label={
              <Box>
                <Typography variant="body2">
                  {prop.key.charAt(0).toUpperCase() + prop.key.slice(1)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {prop.description}
                </Typography>
              </Box>
            }
            sx={{ mt: 2, mb: 1 }}
          />
        );
      case 'select':
        return (
          <FormControl fullWidth margin="normal">
            <InputLabel>{prop.key.charAt(0).toUpperCase() + prop.key.slice(1)}</InputLabel>
            <Select
              value={value || ''}
              label={prop.key.charAt(0).toUpperCase() + prop.key.slice(1)}
              onChange={(e) => handlePropertyChange(prop.key, e.target.value)}
              error={!!error}
            >
              {prop.options?.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
            {(error || prop.description) && (
              <Typography variant="caption" color={error ? 'error' : 'text.secondary'} sx={{ mt: 0.5, ml: 2 }}>
                {error || prop.description}
              </Typography>
            )}
          </FormControl>
        );
      default:
        return null;
    }
  };

  const getNodeIcon = () => {
    switch (nodeType) {
      case 'project': return <Engineering />;
      case 'geometry': return <Science />;
      case 'mesh': return <Speed />;
      case 'file': return <Settings />;
      default: return <Info />;
    }
  };

  const getNodeTypeLabel = () => {
    switch (nodeType) {
      case 'project': return 'Project Properties';
      case 'geometry': return 'Geometry Properties';
      case 'mesh': return 'Mesh Properties';
      case 'file': return 'Load/Constraint Properties';
      default: return 'Node Properties';
    }
  };

  if (!open || !nodeId) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {getNodeIcon()}
          {getNodeTypeLabel()}
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ width: '100%' }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="General" />
            <Tab label="Advanced" />
            <Tab label="History" />
          </Tabs>

          {activeTab === 0 && (
            <Box sx={{ mt: 3 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                  <Typography variant="h6" gutterBottom>Basic Properties</Typography>
                  {nodeProperties.slice(0, Math.ceil(nodeProperties.length / 2)).map((prop) => (
                    <Box key={prop.key}>
                      {renderPropertyInput(prop)}
                    </Box>
                  ))}
                </Grid>
                <Grid item xs={12} md={4}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>Information</Typography>
                      <List dense>
                        <ListItem>
                          <ListItemText
                            primary="Node ID"
                            secondary={nodeId}
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText
                            primary="Type"
                            secondary={nodeType}
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText
                            primary="Status"
                            secondary={
                              <Chip
                                label="Active"
                                size="small"
                                color="success"
                              />
                            }
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText
                            primary="Last Modified"
                            secondary={new Date().toLocaleString()}
                          />
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          )}

          {activeTab === 1 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>Advanced Properties</Typography>
              {nodeProperties.slice(Math.ceil(nodeProperties.length / 2)).map((prop) => (
                <Box key={prop.key}>
                  {renderPropertyInput(prop)}
                </Box>
              ))}

              {nodeType === 'mesh' && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    Changing mesh properties will require regenerating the mesh.
                    This operation may take several minutes depending on the complexity.
                  </Typography>
                </Alert>
              )}
            </Box>
          )}

          {activeTab === 2 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>Change History</Typography>
              <List>
                <ListItem>
                  <ListItemText
                    primary="Node created"
                    secondary="2024-01-15 10:30:00"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Properties modified"
                    secondary="2024-01-15 14:22:00"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Material assignment updated"
                    secondary="2024-01-15 16:45:00"
                  />
                </ListItem>
              </List>
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={handleSave}>
          Save Changes
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default NodeProperties;