use std::sync::*;

use anyhow::*;
use wit_parser::*;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct PackageIdentifier {
    namespace: Arc<str>,
    name: Arc<str>,
    version: Option<semver::Version>,
}

impl PackageIdentifier {
    pub fn new(
        namespace: impl Into<Arc<str>>,
        name: impl Into<Arc<str>>,
        version: Option<semver::Version>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            version,
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> Option<&semver::Version> {
        self.version.as_ref()
    }
}

impl From<&PackageName> for PackageIdentifier {
    fn from(value: &PackageName) -> Self {
        Self {
            namespace: value.namespace.as_str().into(),
            name: value.name.as_str().into(),
            version: value.version.clone(),
        }
    }
}

impl TryFrom<&str> for PackageIdentifier {
    type Error = Error;

    fn try_from(id: &str) -> Result<Self> {
        let colon = id
            .find(':')
            .ok_or_else(|| Error::msg(format!("Expected ':' in identifier {id}")))?;
        let colon_next = colon + 1;
        let namespace = &id[0..colon];
        validate_id(namespace)?;
        let version_index = id[colon_next..].find('@').map(|x| x + colon_next);
        let name = &id[colon_next..version_index.unwrap_or(id.len())];
        validate_id(name)?;

        let version = if let Some(idx) = version_index {
            Some(semver::Version::parse(&id[(idx + 1)..])?)
        } else {
            None
        };

        Ok(Self {
            namespace: namespace.into(),
            name: name.into(),
            version,
        })
    }
}

impl std::fmt::Debug for PackageIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(version) = self.version() {
            f.write_fmt(format_args!(
                "{}:{}@{}",
                self.namespace(),
                self.name(),
                version
            ))
        } else {
            f.write_fmt(format_args!("{}:{}", self.namespace(), self.name()))
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct InterfaceIdentifier {
    package: PackageIdentifier,
    name: Arc<str>,
}

impl InterfaceIdentifier {
    pub fn new(package: PackageIdentifier, name: impl Into<Arc<str>>) -> Self {
        Self {
            package,
            name: name.into(),
        }
    }

    pub fn package(&self) -> &PackageIdentifier {
        &self.package
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl TryFrom<&str> for InterfaceIdentifier {
    type Error = Error;

    fn try_from(id: &str) -> Result<Self> {
        let colon = id
            .find(':')
            .ok_or_else(|| Error::msg(format!("Expected ':' in identifier {id}")))?;
        let colon_next = colon + 1;
        let namespace = &id[0..colon];
        validate_id(namespace)?;
        let interface_index = id[colon_next..]
            .find('/')
            .ok_or_else(|| Error::msg(format!("Expected '/' in identifier {id}")))?
            + colon_next;
        let interface_index_next = interface_index + 1;

        let name = &id[(colon + 1)..interface_index];
        validate_id(name)?;

        let version_index = id[interface_index_next..]
            .find('@')
            .map(|x| x + interface_index_next);
        let interface_name = &id[interface_index_next..version_index.unwrap_or(id.len())];
        validate_id(interface_name)?;

        let version = if let Some(idx) = version_index {
            Some(semver::Version::parse(&id[(idx + 1)..])?)
        } else {
            None
        };

        Ok(Self {
            package: PackageIdentifier::new(namespace, name, version),
            name: interface_name.into(),
        })
    }
}

impl std::fmt::Debug for InterfaceIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(version) = self.package.version() {
            f.write_fmt(format_args!(
                "{}:{}/{}@{}",
                self.package().namespace(),
                self.package().name(),
                self.name(),
                version
            ))
        } else {
            f.write_fmt(format_args!(
                "{}:{}/{}",
                self.package().namespace(),
                self.package().name(),
                self.name()
            ))
        }
    }
}
