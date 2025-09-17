import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
} from '@mui/material';
import { AccountCircle } from '@mui/icons-material';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../services/AuthContext';

export default function Navigation() {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
    handleClose();
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Surrogate Model Platform
        </Typography>

        {isAuthenticated && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Button color="inherit" component={Link} to="/">
              Dashboard
            </Button>
            <Button color="inherit" component={Link} to="/datasets">
              Datasets
            </Button>
            <Button color="inherit" component={Link} to="/models">
              Models
            </Button>
            <Button color="inherit" component={Link} to="/predictions">
              Predictions
            </Button>
            <Button color="inherit" component={Link} to="/nextgen-ml" sx={{
              background: 'linear-gradient(45deg, #FF6B6B, #4ECDC4)',
              borderRadius: '20px',
              px: 2,
              fontWeight: 'bold'
            }}>
              🚀 Next-Gen ML
            </Button>
            <Button color="inherit" component={Link} to="/visualizations" sx={{
              background: 'linear-gradient(45deg, #9C27B0, #E91E63)',
              borderRadius: '20px',
              px: 2,
              fontWeight: 'bold'
            }}>
              📊 Visualizations
            </Button>

            <IconButton
              size="large"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleMenu}
              color="inherit"
            >
              <AccountCircle />
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={Boolean(anchorEl)}
              onClose={handleClose}
            >
              <MenuItem disabled>{user?.email}</MenuItem>
              <MenuItem onClick={handleLogout}>Logout</MenuItem>
            </Menu>
          </Box>
        )}
      </Toolbar>
    </AppBar>
  );
}