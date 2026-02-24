# Documentation Rebuild Instructions

## What Was Fixed

1. **mkdocs.yml Configuration**:
   - Added `griffe_pydantic` extension for proper Pydantic model documentation
   - Added custom `CleanASCIIExtension` to remove ASCII art banners from docstrings
   - Disabled duplicate documentation (class docstring + pydantic fields)
   - Configured filters to hide internal Pydantic methods

2. **CSS Theme (docs/css/extra.css)** - PyTorch-Inspired Clean Design:
   - Minimal, professional styling matching PyTorch documentation
   - Clean tables with simple borders (no gradients or colors)
   - Proper font sizes (base: 15px, code: 13px)
   - GitHub-style colors (borders: #d0d7de, code bg: #f6f8fa)
   - Clear visual separation between classes/modules with horizontal dividers
   - Compact sidebar navigation with reduced padding
   - Better use of horizontal space (max-width: 1400px)
   - Visual hierarchy: Classes are prominent (1.5rem), methods medium (1.1rem), attributes small (1rem)

3. **ASCII Art Cleanup**:
   - Created `docs/griffe_clean_ascii.py` extension to automatically strip ASCII art banners
   - Module docstrings are now clean and professional
   - Updated profile.py as example of clean docstring format

## Installation Steps

Since you're working in the `unaiverse` conda environment, run:

```bash
# Activate your conda environment
conda activate unaiverse

# Install griffe-pydantic
pip install griffe-pydantic

# Rebuild the documentation
mkdocs build --clean

# Or serve it locally to preview
mkdocs serve
```

## What the Changes Do

### griffe-pydantic Extension
This extension properly handles Pydantic models by:
- Extracting field information from `Field()` definitions
- Preventing duplicate attribute listings
- Showing Pydantic validators and constraints
- Clean presentation of model schemas

### New CSS Theme
The new theme provides:
- **Smaller text**: Base font is now 0.875rem (14px) instead of 0.94rem (15px)
- **Cozy colors**: Warm purple accents instead of bright sci-fi colors
- **Soft shadows**: Very subtle (0.04-0.08 opacity) instead of harsh shadows
- **Minimal borders**: Light, barely-there borders
- **Comfortable spacing**: Reduced padding and margins
- **Better readability**: Improved line-height and letter-spacing

## If You Still See Duplicates

If you still see both class docstrings AND Pydantic fields shown separately, you may need to:

1. Check if mkdocs is picking up the right config
2. Clear your browser cache
3. Verify griffe-pydantic is installed: `pip list | grep griffe-pydantic`

## Customization

If you want to further customize the theme colors, edit `docs/css/extra.css`:

- `--doc-accent-primary`: Main accent color (currently purple #7c3aed)
- `--doc-accent-secondary`: Secondary accent (currently cyan #06b6d4)
- Font sizes are in the "LAYOUT & TYPOGRAPHY" section
