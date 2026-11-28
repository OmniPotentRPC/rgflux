// C ABI bridge to rgpot::EAMTablePot for RgpotSurface.
// Compiled with -fvisibility=hidden so these symbols are not exported.
#include "rgpot/EAMTable/EAMTablePot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/nlist/MinimumImage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

void write_error(char *buf, std::size_t cap, const char *msg) {
  if (buf == nullptr || cap == 0) {
    return;
  }
  std::strncpy(buf, msg, cap - 1);
  buf[cap - 1] = '\0';
}

rgpot::EAMTableFormat to_format(int code) {
  switch (code) {
  case 0:
    return rgpot::EAMTableFormat::Funcfl;
  case 1:
    return rgpot::EAMTableFormat::Setfl;
  case 2:
    return rgpot::EAMTableFormat::FinnisSinclair;
  default:
    throw std::invalid_argument("unknown EAMTableFormat code");
  }
}

/// rgpot place() sets q = x * (1/step) + 1 and static_casts q to int before
/// it clamps the bin. A quotient outside the int range is undefined; on this
/// platform the cast collapses to the first cubic, so the linear extension
/// above rhomax is the wrong finite value. No rgpot patch here.
bool place_cast_defined(double x, double step) {
  if (!std::isfinite(x) || !std::isfinite(step) || !(step > 0.0)) {
    return false;
  }
  const double rdx = 1.0 / step;
  if (!std::isfinite(rdx)) {
    return false;
  }
  const double q = x * rdx + 1.0;
  const double hi = static_cast<double>(std::numeric_limits<int>::max()) + 1.0;
  const double lo = static_cast<double>(std::numeric_limits<int>::min());
  return std::isfinite(q) && q < hi && q >= lo;
}

bool finite_positive(double step) {
  return std::isfinite(step) && step > 0.0;
}

/// rgpot's spline stores seven coefficients per knot. The int index is
/// stride * (m - 1) + k with k below the stride. A count past that range
/// is refused before the spline is built.
bool spline_index_fits(int n) {
  constexpr int stride = 7;
  constexpr int last = stride - 1;
  const int limit = (std::numeric_limits<int>::max() - last) / stride;
  return n <= limit + 1;
}

/// Same acceptance as read_eam_table, plus a place() cast that fits in int.
/// A unit density is included: place multiplies by 1/drho, and a drho small
/// enough that this overflows is refused even when rhomax/drho still fits.
void require_grid(int nrho, double drho, int nr, double dr, double cutoff) {
  if (nrho < 5 || nr < 5) {
    throw std::invalid_argument("EAM table arrays require at least 5 knots");
  }
  if (!spline_index_fits(nrho) || !spline_index_fits(nr)) {
    throw std::invalid_argument("EAM spline index does not fit in int");
  }
  if (!finite_positive(dr) || !finite_positive(drho) || !finite_positive(cutoff)) {
    throw std::invalid_argument(
        "EAM dr, drho and cutoff must be finite and positive");
  }
  const double rhomax = static_cast<double>(nrho - 1) * drho;
  const double r_span = std::max(cutoff, static_cast<double>(nr - 1) * dr);
  if (!std::isfinite(rhomax) || !std::isfinite(r_span)) {
    throw std::invalid_argument("EAM grid span is not finite");
  }
  if (!place_cast_defined(r_span, dr) || !place_cast_defined(rhomax, drho) ||
      !place_cast_defined(1.0, drho) || !place_cast_defined(-1.0, drho)) {
    throw std::invalid_argument("EAM grid span does not fit in an int bin index");
  }
}

void require_finite_knots(const std::vector<std::vector<double>> &rows) {
  for (const auto &row : rows) {
    for (double value : row) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("EAM table knots must be finite");
      }
    }
  }
}

void require_density_place(const rgpot::EAMTable &table) {
  for (const auto &row : table.rhor) {
    for (double value : row) {
      const double magnitude = std::abs(value);
      if (!place_cast_defined(magnitude, table.drho) ||
          !place_cast_defined(-magnitude, table.drho)) {
        throw std::invalid_argument(
            "EAM grid span does not fit in an int bin index");
      }
    }
  }
}

void require_table_shape(const rgpot::EAMTable &table) {
  const std::size_t n = table.z.size();
  if (n == 0 || table.frho.size() != n || table.rhor.size() != n * n ||
      table.z2r.size() != n * (n + 1) / 2) {
    throw std::invalid_argument("EAM table array count does not match");
  }
  for (const auto &row : table.frho) {
    if (static_cast<int>(row.size()) != table.nrho) {
      throw std::invalid_argument("EAM table array length does not match");
    }
  }
  for (const auto &row : table.rhor) {
    if (static_cast<int>(row.size()) != table.nr) {
      throw std::invalid_argument("EAM table array length does not match");
    }
  }
  for (const auto &row : table.z2r) {
    if (static_cast<int>(row.size()) != table.nr) {
      throw std::invalid_argument("EAM table array length does not match");
    }
  }
}

/// Index-space slopes of PairEAM::interpolate. The value on p in [0, 1] is
/// a cubic, and its Bernstein coefficients bound that cubic. Constant rows
/// keep the knot magnitude. The bound is at least the true spline value, so
/// a place() cast that fits the bound fits the density a neighbor can
/// contribute.
double max_abs_spline_value(const std::vector<double> &f) {
  const int n = static_cast<int>(f.size());
  double bound = 0.0;
  for (double value : f) {
    bound = std::max(bound, std::abs(value));
  }
  if (n < 5) {
    return bound;
  }
  std::vector<double> slope(static_cast<std::size_t>(n), 0.0);
  slope[0] = f[1] - f[0];
  slope[1] = 0.5 * (f[2] - f[0]);
  slope[static_cast<std::size_t>(n - 2)] =
      0.5 * (f[static_cast<std::size_t>(n - 1)] - f[static_cast<std::size_t>(n - 3)]);
  slope[static_cast<std::size_t>(n - 1)] =
      f[static_cast<std::size_t>(n - 1)] - f[static_cast<std::size_t>(n - 2)];
  for (int m = 3; m <= n - 2; ++m) {
    slope[static_cast<std::size_t>(m - 1)] =
        ((f[static_cast<std::size_t>(m - 3)] - f[static_cast<std::size_t>(m + 1)]) +
         8.0 * (f[static_cast<std::size_t>(m)] - f[static_cast<std::size_t>(m - 2)])) /
        12.0;
  }
  for (int m = 1; m <= n - 1; ++m) {
    const double c5 = slope[static_cast<std::size_t>(m - 1)];
    const double c5n = slope[static_cast<std::size_t>(m)];
    const double left = f[static_cast<std::size_t>(m - 1)];
    const double right = f[static_cast<std::size_t>(m)];
    const double df = right - left;
    const double c4 = 3.0 * df - 2.0 * c5 - c5n;
    const double c3 = c5 + c5n - 2.0 * df;
    const double b0 = left;
    const double b1 = left + c5 / 3.0;
    const double b2 = left + (2.0 / 3.0) * c5 + (1.0 / 3.0) * c4;
    const double b3 = left + c5 + c4 + c3;
    double span = std::max(std::max(std::abs(b0), std::abs(b1)),
                            std::max(std::abs(b2), std::abs(b3)));
    // A few ulps keep the bound above the Horner value the spline evaluates.
    if (std::isfinite(span) && span > 0.0) {
      const double pad = 1.0 + 16.0 * std::numeric_limits<double>::epsilon();
      span = std::nextafter(span * pad, std::numeric_limits<double>::infinity());
    }
    bound = std::max(bound, span);
  }
  return bound;
}

double max_abs_density(const rgpot::EAMTable &table) {
  double bound = 0.0;
  for (const auto &row : table.rhor) {
    bound = std::max(bound, max_abs_spline_value(row));
  }
  if (!std::isfinite(bound)) {
    throw std::invalid_argument("EAM grid span does not fit in an int bin index");
  }
  return bound;
}

struct EamBridge {
  std::unique_ptr<rgpot::EAMTablePot> pot;
  double dr = 0.0;
  double drho = 0.0;
  double cutoff = 0.0;
  double max_abs_density = 0.0;
  bool periodic = false;
};

struct HeaderGrid {
  bool ok = false;
  int nrho = 0;
  int nr = 0;
  double drho = 0.0;
  double dr = 0.0;
  double cutoff = 0.0;
};

int count_from_token(double value) {
  if (!std::isfinite(value) ||
      std::abs(value) > static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("EAM grid span is not finite");
  }
  return static_cast<int>(std::lround(value));
}

/// strtod accepts inf and nan, which iostream extraction on this libstdc++
/// rejects. The grid line is read that way so a non-finite spacing is refused
/// before read_eam_table calls lround.
bool parse_grid_line(const std::string &line, HeaderGrid &grid) {
  std::istringstream in(line);
  std::string word;
  double vals[5];
  for (double &val : vals) {
    if (!(in >> word)) {
      return false;
    }
    char *end = nullptr;
    val = std::strtod(word.c_str(), &end);
    if (end == word.c_str() || *end != '\0') {
      return false;
    }
  }
  grid.nrho = count_from_token(vals[0]);
  grid.drho = vals[1];
  grid.nr = count_from_token(vals[2]);
  grid.dr = vals[3];
  grid.cutoff = vals[4];
  grid.ok = true;
  return true;
}

bool blank_line(const std::string &line) {
  return line.find_first_not_of(" \t\r") == std::string::npos;
}

/// read_eam_table consumes funcfl's one comment line, or the three setfl
/// and Finnis-Sinclair comment lines, with getline, so a blank line counts.
/// The element record and the grid record then use operator>>, which skips
/// blank lines. The grid line is the content line after that element record.
HeaderGrid sniff_grid(const std::string &text, rgpot::EAMTableFormat format) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    lines.push_back(std::move(line));
  }
  HeaderGrid grid;
  std::size_t index = 0;
  auto take = [&](std::string &out) {
    if (index >= lines.size()) {
      return false;
    }
    out = lines[index++];
    return true;
  };
  auto take_content = [&](std::string &out) {
    while (index < lines.size() && blank_line(lines[index])) {
      ++index;
    }
    return take(out);
  };
  const int comments = format == rgpot::EAMTableFormat::Funcfl ? 1 : 3;
  for (int k = 0; k < comments; ++k) {
    if (!take(line)) {
      return grid;
    }
  }
  if (!take_content(line) || !take_content(line)) {
    return grid;
  }
  parse_grid_line(line, grid);
  return grid;
}

void require_finite_numeric_tokens(const std::string &text, rgpot::EAMTableFormat format) {
  std::istringstream in(text);
  std::string line;
  const int comments = format == rgpot::EAMTableFormat::Funcfl ? 1 : 3;
  for (int k = 0; k < comments; ++k) {
    if (!std::getline(in, line)) {
      return;
    }
  }
  std::string word;
  while (in >> word) {
    char *end = nullptr;
    const double value = std::strtod(word.c_str(), &end);
    if (end != word.c_str() && *end == '\0' && !std::isfinite(value)) {
      throw std::invalid_argument("EAM table knots must be finite");
    }
  }
}

void require_header(const HeaderGrid &grid) {
  if (grid.nrho < 5 || grid.nr < 5) {
    throw std::invalid_argument("EAM table arrays require at least 5 knots");
  }
  if (!finite_positive(grid.dr) || !finite_positive(grid.drho) ||
      !finite_positive(grid.cutoff)) {
    throw std::invalid_argument(
        "EAM dr, drho and cutoff must be finite and positive");
  }
  const double rhomax = static_cast<double>(grid.nrho - 1) * grid.drho;
  const double rmax = static_cast<double>(grid.nr - 1) * grid.dr;
  if (!std::isfinite(rhomax) || !std::isfinite(rmax)) {
    throw std::invalid_argument("EAM grid span is not finite");
  }
  const double rho_ratio = rhomax / grid.drho;
  const double r_ratio = rmax / grid.dr;
  const double limit = static_cast<double>(std::numeric_limits<int>::max());
  if (!std::isfinite(rho_ratio) || !std::isfinite(r_ratio) || std::abs(rho_ratio) > limit ||
      std::abs(r_ratio) > limit) {
    throw std::invalid_argument("EAM grid span does not fit in an int bin index");
  }
  require_grid(grid.nrho, grid.drho, grid.nr, grid.dr, grid.cutoff);
}

void adopt(const rgpot::EAMTable &table, bool periodic, double skin, EamBridge &bridge) {
  require_grid(table.nrho, table.drho, table.nr, table.dr, table.cutoff);
  require_finite_knots(table.frho);
  require_finite_knots(table.rhor);
  require_finite_knots(table.z2r);
  require_density_place(table);
  require_table_shape(table);
  bridge.pot = std::make_unique<rgpot::EAMTablePot>(table, periodic, false, skin);
  bridge.dr = table.dr;
  bridge.drho = table.drho;
  bridge.cutoff = table.cutoff;
  bridge.max_abs_density = max_abs_density(table);
  bridge.periodic = periodic;
}

/// Worst-case |rho| is one max-|density| contribution per neighbor inside
/// the cutoff. place() has no density ceiling, so a sum that would overflow
/// the cast is an error. Atoms outside the cutoff do not contribute.
int refuse_overflowing_place(const EamBridge &bridge, std::size_t n_atoms, const double *pos,
                             const double *box9, char *err_buf, std::size_t err_cap) {
  if (!place_cast_defined(bridge.cutoff, bridge.dr)) {
    write_error(err_buf, err_cap, "EAM place index does not fit in int");
    return 2;
  }
  const double cut2 = bridge.cutoff * bridge.cutoff;
  std::optional<rgpot::nlist::MinimumImage> image;
  if (bridge.periodic) {
    image.emplace(box9, std::array<bool, 3>{true, true, true});
  }
  for (std::size_t i = 0; i < n_atoms; ++i) {
    double neighbors = 0.0;
    const double *pi = pos + 3 * i;
    for (std::size_t j = 0; j < n_atoms; ++j) {
      if (i == j) {
        continue;
      }
      const double *pj = pos + 3 * j;
      double dx = pi[0] - pj[0];
      double dy = pi[1] - pj[1];
      double dz = pi[2] - pj[2];
      if (image && !image->foldWithin(dx, dy, dz, cut2)) {
        continue;
      }
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || !(r2 < cut2)) {
        continue;
      }
      neighbors += 1.0;
    }
    const double rho = neighbors * bridge.max_abs_density;
    if (!std::isfinite(rho) || !place_cast_defined(rho, bridge.drho) ||
        !place_cast_defined(-rho, bridge.drho)) {
      write_error(err_buf, err_cap, "EAM place index does not fit in int");
      return 2;
    }
  }
  return 0;
}

} // namespace

extern "C" {

void *rgflux_eam_table_open_file(const char *path, int format, int periodic,
                                 double skin, char *err_buf,
                                 std::size_t err_cap) {
  try {
    if (path == nullptr) {
      throw std::invalid_argument("EAM table path is null");
    }
    std::ifstream in(path);
    if (!in) {
      throw std::invalid_argument(std::string("EAM table: cannot open ") + path);
    }
    std::ostringstream copied;
    copied << in.rdbuf();
    const std::string text = copied.str();
    const auto kind = to_format(format);
    const HeaderGrid header = sniff_grid(text, kind);
    if (header.ok) {
      require_header(header);
    }
    require_finite_numeric_tokens(text, kind);
    std::istringstream table_in(text);
    auto table = rgpot::read_eam_table(table_in, kind);
    if (kind == rgpot::EAMTableFormat::Funcfl && (table.nrho < 5 || table.nr < 5)) {
      throw std::invalid_argument("funcfl resampling left fewer than 5 knots");
    }
    auto bridge = std::make_unique<EamBridge>();
    adopt(table, periodic != 0, skin, *bridge);
    return bridge.release();
  } catch (const std::exception &ex) {
    write_error(err_buf, err_cap, ex.what());
    return nullptr;
  } catch (...) {
    write_error(err_buf, err_cap, "unknown C++ exception in EAMTablePot");
    return nullptr;
  }
}

void *rgflux_eam_table_open_memory(int z, int nrho, double drho, int nr,
                                   double dr, double cutoff,
                                   const double *frho, const double *rhor,
                                   const double *z2r, int periodic, double skin,
                                   char *err_buf, std::size_t err_cap) {
  try {
    if (frho == nullptr || rhor == nullptr || z2r == nullptr) {
      throw std::invalid_argument("invalid EAM table array arguments");
    }
    require_grid(nrho, drho, nr, dr, cutoff);
    rgpot::EAMTable table;
    table.nrho = nrho;
    table.drho = drho;
    table.nr = nr;
    table.dr = dr;
    table.cutoff = cutoff;
    table.rhomax = static_cast<double>(nrho - 1) * drho;
    table.z = {z};
    table.frho = {std::vector<double>(frho, frho + nrho)};
    table.rhor = {std::vector<double>(rhor, rhor + nr)};
    table.z2r = {std::vector<double>(z2r, z2r + nr)};
    auto bridge = std::make_unique<EamBridge>();
    adopt(table, periodic != 0, skin, *bridge);
    return bridge.release();
  } catch (const std::exception &ex) {
    write_error(err_buf, err_cap, ex.what());
    return nullptr;
  } catch (...) {
    write_error(err_buf, err_cap, "unknown C++ exception in EAMTablePot");
    return nullptr;
  }
}

void rgflux_eam_table_close(void *handle) {
  delete static_cast<EamBridge *>(handle);
}

int rgflux_eam_table_eval(const void *handle, std::size_t n_atoms,
                          const double *pos, const int *atmnrs,
                          const double *box9, double *energy, double *forces,
                          char *err_buf, std::size_t err_cap) {
  if (handle == nullptr || pos == nullptr || atmnrs == nullptr ||
      box9 == nullptr || energy == nullptr || forces == nullptr) {
    write_error(err_buf, err_cap, "null buffer passed to rgflux_eam_table_eval");
    return 1;
  }
  try {
    const auto *bridge = static_cast<const EamBridge *>(handle);
    if (const int rc = refuse_overflowing_place(*bridge, n_atoms, pos, box9, err_buf, err_cap)) {
      return rc;
    }
    rgpot::ForceInput input{
        .nAtoms = n_atoms, .pos = pos, .atmnrs = atmnrs, .box = box9};
    rgpot::ForceOut output{.F = forces,
                           .energy = 0.0,
                           .variance = 0.0,
                           .stress = {},
                           .has_stress = 0};
    bridge->pot->forceImpl(input, &output);
    *energy = output.energy;
    return 0;
  } catch (const std::exception &ex) {
    write_error(err_buf, err_cap, ex.what());
    return 2;
  } catch (...) {
    write_error(err_buf, err_cap, "unknown C++ exception in EAMTablePot::forceImpl");
    return 3;
  }
}

} // extern "C"
